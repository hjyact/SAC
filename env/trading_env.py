"""
env/trading_env.py — Gymnasium 트레이딩 환경

설계 원칙:
  - 연속 행동 공간 (포지션 비율 [-1, 1])
      -1: 최대 공매도, 0: 현금, +1: 최대 매수
      → SAC의 연속 행동 공간 활용 극대화
      → 포지션 조정 = action - current_position (거래 발생 시에만 수수료)

  - 보상 설계 (Reward Shaping):
      1. PnL 보상: 단순 수익률
      2. Sharpe 보상: 위험 조정 수익 (rolling Sharpe)
      3. Sortino 보상: 하방 위험만 패널티
      4. Mixed: 위 조합 + MDD 패널티 + 거래비용 명시적 반영

  - 관측 공간:
      [window_size × n_features] (기술적 지표 히스토리)
      + [4] 포트폴리오 상태 (position, unrealized_pnl, cash_ratio, holding_time)

  - Look-ahead Bias 방지:
      step()에서는 현재까지의 데이터만 사용
      다음 종가는 action 적용 후에만 접근
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Optional, Any
import warnings
warnings.filterwarnings("ignore")

from config import EnvConfig, env_cfg
from utils.features import compute_portfolio_features


class TradingEnv(gym.Env):
    """
    연속 행동 공간 SAC 트레이딩 환경.

    Observation: (window_size × n_features + 4,) flat vector
    Action:      (1,) ∈ [-1, 1]  →  목표 포지션 비율
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        feature_df: pd.DataFrame,
        price_df: pd.DataFrame,
        cfg: EnvConfig = env_cfg,
        mode: str = "train",   # "train" | "eval"
        norm_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """
        Parameters
        ----------
        feature_df : build_all_features()로 만든 피처 DataFrame (NaN 제거 완료)
        price_df   : 원본 OHLCV (체결가 계산용)
        cfg        : EnvConfig
        mode       : train → 랜덤 시작점, eval → 처음부터
        """
        super().__init__()
        self.cfg  = cfg
        self.mode = mode

        # 데이터 정렬
        common_idx   = feature_df.index.intersection(price_df.index)
        # forward-fill로 NaN 처리 (초기 윈도우 기간만 NaN 존재)
        feat_filled  = feature_df.loc[common_idx].ffill().bfill()
        self.features = feat_filled.values.astype(np.float32)
        self.prices   = price_df.loc[common_idx].reset_index(drop=True)
        self.n_steps  = len(self.features)
        self.n_feat   = self.features.shape[1]

        # 피처 정규화 통계: 외부 주입(훈련 env 기준)이 없으면 자체 계산
        if norm_stats is not None:
            self._feat_mean, self._feat_std = norm_stats
        else:
            self._feat_mean = np.nanmean(self.features, axis=0)
            self._feat_std  = np.nanstd(self.features, axis=0) + 1e-8

        # 공간 정의
        obs_dim = cfg.window_size * self.n_feat + 4   # 4 = 포트폴리오 상태
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
        # 연속 행동: [-1, 1] 목표 포지션 (0=현금, 1=풀매수, -1=풀공매도)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # 내부 상태
        self._reset_state()

        # 보상 계산용 히스토리 버퍼
        self._ret_history: list = []

    @property
    def norm_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """훈련 데이터 기반 정규화 통계 (eval env에 주입용)."""
        return (self._feat_mean.copy(), self._feat_std.copy())

    # ── 리셋 ──────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._reset_state()

        # 에피소드 시작점
        min_start = self.cfg.window_size
        if self.mode == "train" and self.cfg.use_random_start:
            max_start = max(min_start + 1, self.n_steps - self.cfg.episode_length - 1)
            self._start = self.np_random.integers(min_start, max_start)
        else:
            self._start = min_start

        self._step_idx = self._start
        self._end      = min(self._start + self.cfg.episode_length, self.n_steps - 1)
        self._episode_start_price = float(self.prices.iloc[self._start]["Close"])

        obs = self._get_obs()
        return obs, {}

    def _reset_state(self):
        self._start       = self.cfg.window_size
        self._step_idx    = self._start
        self._end         = self.n_steps - 1
        self.capital      = self.cfg.initial_capital
        self.position     = 0.0      # [-1, 1] 현재 포지션
        self.cash         = self.cfg.initial_capital
        self.holdings     = 0.0      # 보유 주식 금액
        self.entry_price  = 0.0
        self.holding_steps = 0
        self.peak_capital = self.cfg.initial_capital
        self._ret_history = []
        self._trade_count = 0
        self._episode_start_price = 0.0

    # ── 스텝 ──────────────────────────────────────────

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        action: ndarray shape (1,), 목표 포지션 비율 ∈ [-1, 1]

        순서:
          1. 목표 포지션 결정
          2. 현재→목표 포지션 조정 (수수료 부과)
          3. 다음 봉 종가로 PnL 계산
          4. 보상 계산
          5. 다음 관측 반환
        """
        target_pos = float(np.clip(action[0], -1.0, 1.0))
        target_pos = np.clip(target_pos, -self.cfg.max_position, self.cfg.max_position)

        # 현재 시점 가격 (체결가 = 현재 봉 종가 + 슬리피지)
        cur_close  = float(self.prices.iloc[self._step_idx]["Close"])
        exec_price = cur_close * (1 + np.sign(target_pos - self.position) * self.cfg.slippage)

        # 포지션 변화량
        pos_delta  = target_pos - self.position
        trade_cost = abs(pos_delta) * self.capital * self.cfg.commission

        # 포지션 업데이트
        self.position = target_pos
        self.cash     = self.capital * (1 - target_pos) - trade_cost
        self.holdings = self.capital * target_pos

        if abs(pos_delta) > 0.01:
            self._trade_count += 1
            if target_pos != 0:
                self.entry_price = exec_price
            self.holding_steps = 0
        else:
            self.holding_steps += 1

        # 다음 봉 종가로 PnL
        self._step_idx += 1
        next_close = float(self.prices.iloc[self._step_idx]["Close"])
        price_ret  = (next_close - cur_close) / (cur_close + 1e-9)

        # 포지션 수익 (롱: 상승 이익, 숏: 하락 이익)
        pre_capital  = self.capital                          # PnL 계산 전 자본 저장
        position_pnl = self.position * price_ret * pre_capital
        self.capital  = max(pre_capital + position_pnl - trade_cost, 1.0)
        self.holdings = self.capital * abs(self.position)

        # 최고점 업데이트 (MDD 계산용)
        self.peak_capital = max(self.peak_capital, self.capital)

        # step_ret: 업데이트 전 자본 대비 수익률 (올바른 분모)
        step_ret = position_pnl / (pre_capital + 1e-9)
        self._ret_history.append(step_ret)
        reward = self._compute_reward(step_ret, trade_cost)

        # 종료 조건
        terminated = self.capital <= self.cfg.initial_capital * 0.5  # 50% 손실
        truncated  = self._step_idx >= self._end

        obs  = self._get_obs()
        info = self._get_info(step_ret, trade_cost, next_close)

        return obs, reward, terminated, truncated, info

    # ── 보상 함수 ──────────────────────────────────────

    def _compute_reward(self, step_ret: float, trade_cost: float) -> float:
        """
        보상 설계 원칙:
          - 모든 보상은 [-5, +5] 범위 내로 클리핑
          - step_ret은 이미 현재 자본 대비 수익률 (소수점 단위)
          - Sharpe 계산: 최근 20 스텝 롤링 (너무 길면 gradient signal 약해짐)
          - 거래비용 패널티: 실제 cost를 자본 대비 비율로 정규화
        """
        cfg = self.cfg

        # 항상 수행: 거래비용 패널티 (포지션 변동 시에만 유의미)
        cost_norm = trade_cost / (self.capital + 1e-9)  # 자본 대비 비율

        if cfg.reward_type == "pnl":
            # 단순 PnL: 수익률 × 스케일 (거래비용 명시 차감)
            reward = (step_ret - cost_norm) * cfg.reward_scaling
            return float(np.clip(reward, -5.0, 5.0))

        rets = np.array(self._ret_history[-20:])  # 최근 20 스텝
        if len(rets) < 5:
            # 버퍼 부족 시 단순 PnL
            return float(np.clip(step_ret * cfg.reward_scaling, -5.0, 5.0))

        mean_ret = rets.mean()
        std_ret  = rets.std() + 1e-8

        if cfg.reward_type == "sharpe":
            # 롤링 Sharpe (일별 기준, 연율화 제거 → 스케일 폭발 방지)
            sharpe = mean_ret / std_ret
            reward = sharpe * cfg.reward_scaling - cost_norm * 10.0
            return float(np.clip(reward, -5.0, 5.0))

        elif cfg.reward_type == "sortino":
            # Sortino: 하방 위험만 패널티
            downside = rets[rets < 0]
            down_std = downside.std() + 1e-8 if len(downside) > 0 else std_ret
            sortino  = mean_ret / down_std
            reward   = sortino * cfg.reward_scaling - cost_norm * 10.0
            return float(np.clip(reward, -5.0, 5.0))

        elif cfg.reward_type == "mixed":
            # ① Sharpe 성분
            sharpe = mean_ret / std_ret

            # ② MDD 패널티 (현재 자본 기준)
            mdd_ratio = (self.capital - self.peak_capital) / (self.peak_capital + 1e-9)
            mdd_pen   = cfg.drawdown_penalty * min(mdd_ratio, 0.0)  # 항상 ≤ 0

            # ③ 거래비용 패널티 (과도한 매매 억제)
            cost_pen = -cfg.risk_penalty * cost_norm * 100.0

            # ④ 과도한 레버리지 패널티 (|pos| > 0.8 구간에서 급증)
            leverage_pen = -0.02 * max(0.0, abs(self.position) - 0.8) ** 2

            reward = (
                sharpe * cfg.reward_scaling
                + mdd_pen
                + cost_pen
                + leverage_pen
            )
            return float(np.clip(reward, -5.0, 5.0))

        return float(np.clip(step_ret * cfg.reward_scaling, -5.0, 5.0))

    # ── 관측 ──────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """
        [window_size × n_feat] 피처 히스토리 + 포트폴리오 상태 4개
        → flat vector
        """
        start = self._step_idx - self.cfg.window_size
        end   = self._step_idx

        window = self.features[start:end]  # (window_size, n_feat)

        # 표준화
        window_norm = (window - self._feat_mean) / self._feat_std
        window_norm = np.clip(window_norm, -10.0, 10.0)
        window_norm = np.nan_to_num(window_norm, nan=0.0)

        # 포트폴리오 상태
        cur_close        = float(self.prices.iloc[self._step_idx]["Close"])
        unrealized_pnl   = (cur_close - self.entry_price) / (self.entry_price + 1e-9) if self.entry_price > 0 else 0.0
        cash_ratio       = max(self.cash, 0) / (self.capital + 1e-9)

        port_feat = compute_portfolio_features(
            position=self.position,
            unrealized_pnl_pct=unrealized_pnl,
            cash_ratio=cash_ratio,
            holding_steps=self.holding_steps,
        )

        obs = np.concatenate([window_norm.flatten(), port_feat])
        return obs.astype(np.float32)

    def _get_info(self, step_ret, trade_cost, cur_price) -> Dict:
        total_ret = (self.capital - self.cfg.initial_capital) / self.cfg.initial_capital
        mdd = (self.capital - self.peak_capital) / (self.peak_capital + 1e-9)
        return {
            "capital":    self.capital,
            "position":   self.position,
            "step_ret":   step_ret,
            "total_ret":  total_ret,
            "mdd":        mdd,
            "trade_cost": trade_cost,
            "trade_count": self._trade_count,
            "cur_price":  cur_price,
        }

    def get_buyhold_return(self) -> float:
        """현재 에피소드 기간의 단순 매수보유 수익률."""
        cur_price = float(self.prices.iloc[self._step_idx]["Close"])
        return (cur_price - self._episode_start_price) / (self._episode_start_price + 1e-9)

    # ── 렌더링 ─────────────────────────────────────────

    def render(self):
        total_ret = (self.capital - self.cfg.initial_capital) / self.cfg.initial_capital
        mdd = (self.capital - self.peak_capital) / (self.peak_capital + 1e-9)
        print(
            f"Step {self._step_idx:4d} | "
            f"Capital: {self.capital:>12,.0f} | "
            f"Ret: {total_ret:>+7.2%} | "
            f"Pos: {self.position:>+5.2f} | "
            f"MDD: {mdd:>7.2%} | "
            f"Trades: {self._trade_count}"
        )
