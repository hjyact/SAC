"""
training/trainer.py — SAC 학습 루프

학습 전략:
  - Warm-up: 초기 N 스텝은 랜덤 행동으로 버퍼 채움
  - Off-policy: 환경 1 스텝 → gradient 1회 (gradient_steps 설정 가능)
  - Periodic Eval: eval_interval마다 결정론적 정책으로 평가
  - Early Stopping: eval Sharpe 기준
  - Curriculum: 쉬운 에피소드 → 어려운 에피소드 (선택적)
"""

import numpy as np
import torch
import logging
import time
from pathlib import Path
from collections import deque
from typing import Optional, Dict, List

from config import TrainConfig, EnvConfig, SACConfig, train_cfg, RESULT_DIR, DEVICE
from agent.sac_agent import SACAgent
from agent.replay_buffer import ReplayBuffer
from env.trading_env import TradingEnv

logger = logging.getLogger(__name__)


class SACTrainer:
    """
    SAC 학습 매니저.
    """

    def __init__(
        self,
        train_env: TradingEnv,
        eval_env:  TradingEnv,
        agent: SACAgent,
        buffer: ReplayBuffer,
        cfg: TrainConfig = train_cfg,
    ):
        self.train_env = train_env
        self.eval_env  = eval_env
        self.agent     = agent
        self.buffer    = buffer
        self.cfg       = cfg

        # 로그
        self._ep_returns: deque = deque(maxlen=50)
        self._ep_lengths: deque = deque(maxlen=50)
        self._loss_history: List[Dict] = []
        self._eval_history: List[Dict] = []

        self._best_eval_score  = -np.inf
        self._patience_counter = 0
        self._timestep         = 0
        self._episode          = 0

    # ── 메인 학습 루프 ─────────────────────────────────

    def train(self) -> List[Dict]:
        """
        전체 학습 루프.

        Returns
        -------
        eval_history : 에포크별 평가 결과 리스트
        """
        logger.info(f"SAC 학습 시작 | device={self.agent.device} | "
                    f"total_steps={self.cfg.total_timesteps:,}")

        obs, _ = self.train_env.reset(seed=self.cfg.seed)
        ep_ret = 0.0
        ep_len = 0
        t_start = time.time()

        for t in range(1, self.cfg.total_timesteps + 1):
            self._timestep = t

            # ── 행동 선택
            if t < self.agent.cfg.min_replay_size:
                # 워밍업: 랜덤 탐험
                action = self.train_env.action_space.sample()
            else:
                action = self.agent.select_action(obs, deterministic=False)

            # ── 환경 스텝
            next_obs, reward, terminated, truncated, info = self.train_env.step(action)
            done = terminated or truncated

            # 버퍼 저장 (terminated ≠ truncated 구분: time-limit은 done=False)
            self.buffer.add(obs, action, reward, next_obs, float(terminated))

            obs     = next_obs
            ep_ret += reward
            ep_len += 1

            # ── 에피소드 종료
            if done:
                self._ep_returns.append(ep_ret)
                self._ep_lengths.append(ep_len)
                self._episode += 1
                obs, _ = self.train_env.reset()
                ep_ret = 0.0
                ep_len = 0

            # ── Gradient 업데이트
            if t >= self.agent.cfg.min_replay_size and self.buffer.is_ready:
                for _ in range(self.agent.cfg.gradient_steps):
                    losses = self.agent.train_step(self.buffer)
                    self._loss_history.append({**losses, "step": t})

            # ── 로깅
            if t % self.cfg.log_interval == 0 and self._ep_returns:
                elapsed = time.time() - t_start
                fps     = t / elapsed
                mean_ret = np.mean(self._ep_returns)
                mean_len = np.mean(self._ep_lengths)
                recent_losses = self._loss_history[-10:] if self._loss_history else [{}]
                mean_critic = np.mean([l.get("critic_loss", 0) for l in recent_losses])
                mean_actor  = np.mean([l.get("actor_loss", 0)  for l in recent_losses])

                logger.info(
                    f"[{t:7,d}/{self.cfg.total_timesteps:,}] "
                    f"Ep={self._episode:4d} | "
                    f"Ret={mean_ret:+7.4f} | "
                    f"Len={mean_len:5.0f} | "
                    f"α={self.agent.alpha:.4f} | "
                    f"Q_loss={mean_critic:.4f} | "
                    f"π_loss={mean_actor:.4f} | "
                    f"FPS={fps:.0f}"
                )

            # ── 평가
            if t % self.cfg.eval_interval == 0:
                eval_result = self._evaluate()
                eval_result["step"] = t
                self._eval_history.append(eval_result)

                bh = eval_result.get("buyhold_return", 0.0)
                logger.info(
                    f"  ── EVAL ── "
                    f"TotalRet={eval_result['total_return']:+.2%} | "
                    f"B&H={bh:+.2%} | "
                    f"Sharpe={eval_result['sharpe']:.3f} | "
                    f"Sortino={eval_result.get('sortino', 0):.3f} | "
                    f"MDD={eval_result['mdd']:.2%} | "
                    f"WinRate={eval_result.get('win_rate', 0):.1%} | "
                    f"Trades={eval_result['mean_trades']:.0f}"
                )

                # 최고 모델 저장 + early stopping
                score = eval_result["sharpe"]
                if score > self._best_eval_score:
                    self._best_eval_score  = score
                    self._patience_counter = 0
                    self.agent.save("best_sac")
                    logger.info(f"  ✅ 최고 모델 저장 (Sharpe={score:.4f})")
                else:
                    self._patience_counter += 1
                    patience = self.cfg.early_stopping_patience
                    if patience > 0 and self._patience_counter >= patience:
                        logger.info(
                            f"  ⏹ Early stopping: {patience}회 연속 개선 없음 "
                            f"(최고 Sharpe={self._best_eval_score:.4f})"
                        )
                        break

            # ── 주기적 저장
            if t % self.cfg.save_interval == 0:
                self.agent.save(f"sac_step_{t}")

        logger.info(f"\n학습 완료 | 최고 Sharpe: {self._best_eval_score:.4f}")
        return self._eval_history

    # ── 평가 ──────────────────────────────────────────

    def _evaluate(self) -> Dict:
        """
        결정론적 정책으로 eval_episodes 에피소드 평가.
        Sharpe는 실제 수익률(step_ret) 기반으로 일관되게 계산.
        """
        all_step_rets = []
        ep_capitals   = []
        ep_mdds       = []
        ep_trades     = []
        ep_bh_rets    = []

        for ep in range(self.cfg.eval_episodes):
            obs, _ = self.eval_env.reset()
            done   = False
            step_rets = []

            while not done:
                action = self.agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                step_rets.append(info["step_ret"])
                done = terminated or truncated

            all_step_rets.extend(step_rets)
            ep_capitals.append(info["capital"])
            ep_mdds.append(info["mdd"])
            ep_trades.append(info["trade_count"])
            ep_bh_rets.append(self.eval_env.get_buyhold_return())

        # 에피소드 평균 총 수익률
        init = self.eval_env.cfg.initial_capital
        mean_capital = np.mean(ep_capitals)
        total_return = (mean_capital - init) / init

        # Sharpe: 전 에피소드 수익률 시계열 기반 (연율화)
        rets   = np.array(all_step_rets)
        sharpe = (rets.mean() / (rets.std() + 1e-8)) * np.sqrt(252) if len(rets) > 1 else 0.0

        # Sortino 추가
        downside    = rets[rets < 0]
        down_std    = downside.std() + 1e-8 if len(downside) > 0 else rets.std() + 1e-8
        sortino     = (rets.mean() / down_std) * np.sqrt(252) if len(rets) > 1 else 0.0

        return {
            "mean_return":    float(rets.mean()),
            "total_return":   float(total_return),
            "sharpe":         float(sharpe),
            "sortino":        float(sortino),
            "mdd":            float(np.mean(ep_mdds)),
            "mean_trades":    float(np.mean(ep_trades)),
            "win_rate":       float((rets > 0).mean()),
            "buyhold_return": float(np.mean(ep_bh_rets)),
        }

    # ── 결과 저장 ──────────────────────────────────────

    def save_results(self):
        import json
        path = RESULT_DIR / "training_log.json"
        with open(path, "w") as f:
            json.dump({
                "eval_history": self._eval_history,
                "loss_history": self._loss_history[-1000:],  # 최근 1000개만
            }, f, indent=2)
        logger.info(f"학습 결과 저장: {path}")
