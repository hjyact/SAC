"""
training/walk_forward.py — Walk-Forward 재학습 시스템

시계열의 비정상성(non-stationarity) 대응.
시장 구조가 변하면 모델도 재학습해야 합니다.

구조:
  ┌──────────┬──────────┬──────────┬──────────┐
  │ Train 1  │  Val 1   │          │          │
  └──────────┴──────────┼──────────┬──────────┤
             │ Train 2  │  Val 2   │          │
             └──────────┴──────────┼──────────┤
                        │ Train 3  │  Val 3   │
                        └──────────┴──────────┘
  → 각 윈도우마다 에이전트 재학습 후 다음 기간 예측
  → 최종 결과: 전체 기간 OOS (Out-of-Sample) 성과
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class WalkForwardTrainer:
    """
    Walk-Forward 방식 재학습 및 평가.

    Parameters
    ----------
    n_splits    : 분할 수
    train_ratio : 훈련 기간 비율 (val 기간 = 1 - train_ratio)
    retrain_from_scratch : True=매 폴드 초기화, False=이전 가중치 계승(warm start)
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_ratio: float = 0.7,
        n_steps_per_fold: int = 50_000,
        retrain_from_scratch: bool = False,
    ):
        self.n_splits      = n_splits
        self.train_ratio   = train_ratio
        self.n_steps       = n_steps_per_fold
        self.from_scratch  = retrain_from_scratch
        self.fold_results: List[Dict] = []

    def run(
        self,
        feat_df: pd.DataFrame,
        price_df: pd.DataFrame,
        env_cfg,
        sac_cfg,
    ) -> Dict:
        """
        전체 Walk-Forward 학습/평가를 수행합니다.

        Returns
        -------
        dict: fold별 결과 + 전체 OOS 성과
        """
        from env.trading_env import TradingEnv
        from agent.sac_agent import SACAgent
        from agent.replay_buffer import ReplayBuffer

        n = len(feat_df)
        fold_size = n // self.n_splits

        all_oos_rets = []
        all_oos_caps = []
        prev_agent   = None

        for fold in range(self.n_splits - 1):
            train_end = fold_size * (fold + 1)
            val_end   = min(fold_size * (fold + 2), n)

            tr_feat  = feat_df.iloc[:train_end]
            tr_price = price_df.iloc[:train_end]
            val_feat  = feat_df.iloc[train_end:val_end]
            val_price = price_df.iloc[train_end:val_end]

            if len(val_feat) < env_cfg.window_size + 10:
                continue

            logger.info(
                f"\n[Fold {fold+1}/{self.n_splits-1}] "
                f"Train: {tr_price.index[0].date()}~{tr_price.index[-1].date()} "
                f"({len(tr_feat)}행) | "
                f"Val: {val_price.index[0].date()}~{val_price.index[-1].date()} "
                f"({len(val_feat)}행)"
            )

            train_env = TradingEnv(tr_feat,  tr_price,  env_cfg, "train")
            val_env   = TradingEnv(val_feat, val_price, env_cfg, "eval",
                                   norm_stats=train_env.norm_stats)

            obs_dim = train_env.observation_space.shape[0]
            act_dim = train_env.action_space.shape[0]

            # 재학습 전략
            if self.from_scratch or prev_agent is None:
                agent = SACAgent(obs_dim, act_dim, sac_cfg)
            else:
                # Warm start: 이전 폴드 가중치 계승 (시장 연속성 활용)
                agent = prev_agent
                logger.info("  Warm start: 이전 폴드 가중치 계승")

            buffer = ReplayBuffer(obs_dim, act_dim, sac_cfg.buffer_size)

            # 학습
            obs, _ = train_env.reset()
            for t in range(self.n_steps):
                if t < sac_cfg.min_replay_size:
                    action = train_env.action_space.sample()
                else:
                    action = agent.select_action(obs, deterministic=False)
                next_obs, r, term, trunc, _ = train_env.step(action)
                buffer.add(obs, action, r, next_obs, float(term))
                obs = next_obs
                if term or trunc:
                    obs, _ = train_env.reset()
                if t >= sac_cfg.min_replay_size and buffer.is_ready:
                    agent.train_step(buffer)

            # OOS 평가
            oos_rets = []
            obs, _ = val_env.reset()
            done = False
            while not done:
                action = agent.select_action(obs, deterministic=True)
                obs, _, term, trunc, info = val_env.step(action)
                oos_rets.append(info["step_ret"])
                done = term or trunc

            oos_rets = np.array(oos_rets)
            sharpe = (oos_rets.mean() / (oos_rets.std() + 1e-8)) * np.sqrt(252)
            total_ret = (info["capital"] - env_cfg.initial_capital) / env_cfg.initial_capital

            fold_result = {
                "fold":       fold + 1,
                "sharpe":     float(sharpe),
                "total_ret":  float(total_ret),
                "mdd":        float(info["mdd"]),
                "n_trades":   info["trade_count"],
                "val_start":  str(val_price.index[0].date()),
                "val_end":    str(val_price.index[-1].date()),
            }
            self.fold_results.append(fold_result)
            all_oos_rets.extend(oos_rets.tolist())
            all_oos_caps.append(info["capital"])

            logger.info(
                f"  OOS 결과: Sharpe={sharpe:.3f} | "
                f"TotalRet={total_ret:+.2%} | "
                f"MDD={info['mdd']:.2%} | "
                f"Trades={info['trade_count']}"
            )

            agent.save(f"wf_fold_{fold+1}")
            prev_agent = agent

        # 전체 OOS 요약
        all_rets = np.array(all_oos_rets)
        summary  = {
            "n_folds":      len(self.fold_results),
            "mean_sharpe":  np.mean([r["sharpe"] for r in self.fold_results]),
            "std_sharpe":   np.std([r["sharpe"] for r in self.fold_results]),
            "mean_ret":     np.mean([r["total_ret"] for r in self.fold_results]),
            "oos_sharpe":   float((all_rets.mean() / (all_rets.std()+1e-8)) * np.sqrt(252)),
            "oos_win_rate": float((all_rets > 0).mean()),
            "folds":        self.fold_results,
        }

        logger.info(f"\n{'='*55}")
        logger.info("Walk-Forward 전체 OOS 요약")
        logger.info(f"{'='*55}")
        logger.info(f"  폴드 평균 Sharpe : {summary['mean_sharpe']:.3f} ± {summary['std_sharpe']:.3f}")
        logger.info(f"  전체 OOS Sharpe  : {summary['oos_sharpe']:.3f}")
        logger.info(f"  전체 OOS 승률    : {summary['oos_win_rate']:.2%}")

        return summary
