"""
agent/sac_agent.py — SAC (Soft Actor-Critic) 에이전트

알고리즘: Haarnoja et al. "Soft Actor-Critic: Off-Policy Maximum Entropy
           Deep Reinforcement Learning with a Stochastic Actor" (2018)
           + SAC v2: "Soft Actor-Critic Algorithms and Applications" (2019)

핵심 수식:
  ① Critic 손실 (Bellman backup + 엔트로피):
        y = r + γ(1-d) · [min_Q(s',ã') - α·log π(ã'|s')]
        L_Q = E[(Q(s,a) - y)²]   (Twin Q 각각)

  ② Actor 손실 (정책 개선):
        L_π = E[α·log π(a|s) - Q(s,a)]
        → 엔트로피를 최대화하면서 Q값도 최대화

  ③ 온도(α) 자동 조정 (SAC v2):
        L_α = E[-α · (log π(a|s) + H̄)]
        H̄: 목표 엔트로피 (보통 -action_dim)
        → α가 자동으로 탐험-활용 균형 조절

  ④ Soft target update:
        θ_target ← τ·θ + (1-τ)·θ_target
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from config import SACConfig, sac_cfg, DEVICE, CKPT_DIR
from networks.sac_nets import GaussianActor, TwinCritic
from agent.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class SACAgent:
    """
    Soft Actor-Critic 에이전트.

    외부 인터페이스:
        select_action(obs, deterministic)  → 행동 선택
        train_step(buffer)                 → 1회 gradient 업데이트
        save(name) / load(name)            → 체크포인트
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        cfg: SACConfig = sac_cfg,
        device: str = DEVICE,
    ):
        self.cfg        = cfg
        self.device     = device
        self.action_dim = action_dim
        self._total_updates = 0

        # ── 네트워크 초기화
        self.actor = GaussianActor(
            obs_dim, action_dim, cfg.hidden_dims, cfg.activation
        ).to(device)

        self.critic = TwinCritic(
            obs_dim, action_dim, cfg.hidden_dims, cfg.activation
        ).to(device)

        self.critic_target = TwinCritic(
            obs_dim, action_dim, cfg.hidden_dims, cfg.activation
        ).to(device)

        # target 네트워크는 critic과 동일하게 초기화 후 고정 (soft update만)
        self._hard_update(self.critic_target, self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # ── 옵티마이저
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        # ── 엔트로피 온도 α (SAC v2 자동 조정)
        if cfg.auto_alpha:
            self.log_alpha   = torch.tensor(
                np.log(cfg.alpha), dtype=torch.float32,
                device=device, requires_grad=True
            )
            self.alpha_opt   = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
            self.target_entropy = cfg.target_entropy if cfg.target_entropy != -1.0 \
                                  else -float(action_dim)
        else:
            self.log_alpha  = torch.tensor(np.log(cfg.alpha), device=device)

        # 로그
        self._loss_log = {"critic": [], "actor": [], "alpha": [], "alpha_val": []}

    # ── 행동 선택 ──────────────────────────────────────

    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """
        obs: (obs_dim,) numpy array
        deterministic: True → 평가 시 결정론적 행동 (탐험 없음)
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        if deterministic:
            action = self.actor.get_deterministic_action(obs_t)
        else:
            action = self.actor.get_action(obs_t)

        return action.flatten()

    # ── 학습 스텝 ──────────────────────────────────────

    def train_step(self, buffer: ReplayBuffer) -> Dict[str, float]:
        """
        리플레이 버퍼에서 배치를 샘플링하고 SAC 업데이트 1회 수행.

        순서:
          1. Critic 업데이트 (Bellman + 엔트로피 target)
          2. Actor 업데이트 (정책 개선)
          3. α 업데이트 (엔트로피 자동 조정)
          4. Soft target update
        """
        batch = buffer.sample(self.cfg.batch_size)
        obs, actions, rewards, next_obs, dones = [
            torch.FloatTensor(b).to(self.device) for b in batch
        ]

        alpha = self.log_alpha.exp().detach()

        # ── ① Critic 업데이트
        critic_loss = self._update_critic(obs, actions, rewards, next_obs, dones, alpha)

        # ── ② Actor 업데이트
        actor_loss = self._update_actor(obs, alpha)

        # ── ③ α 자동 조정
        alpha_loss = 0.0
        if self.cfg.auto_alpha:
            alpha_loss = self._update_alpha(obs)

        # ── ④ Soft target update
        self._total_updates += 1
        if self._total_updates % self.cfg.target_update_interval == 0:
            self._soft_update(self.critic_target, self.critic, self.cfg.tau)

        return {
            "critic_loss": critic_loss,
            "actor_loss":  actor_loss,
            "alpha_loss":  alpha_loss,
            "alpha":       float(self.log_alpha.exp().item()),
        }

    # ── ① Critic 손실 ─────────────────────────────────

    def _update_critic(self, obs, actions, rewards, next_obs, dones, alpha) -> float:
        """
        Bellman backup with entropy regularization:
            y = r + γ(1-d)[min_Q(s',ã') - α·log π(ã'|s')]
        """
        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_obs)
            q_next = self.critic_target.q_min(next_obs, next_action)
            # 엔트로피 보너스: soft Bellman target
            target_q = rewards + self.cfg.gamma * (1 - dones) * (
                q_next - alpha * next_log_prob
            )

        q1, q2   = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        self.critic_opt.step()

        return float(critic_loss.item())

    # ── ② Actor 손실 ──────────────────────────────────

    def _update_actor(self, obs, alpha) -> float:
        """
        정책 개선:
            L_π = E[α·log π(a|s) - min_Q(s,a)]
        Critic은 고정 (actor gradient만)
        """
        action, log_prob = self.actor(obs)
        q_val = self.critic.q_min(obs, action)

        actor_loss = (alpha * log_prob - q_val).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
        self.actor_opt.step()

        return float(actor_loss.item())

    # ── ③ α 자동 조정 ─────────────────────────────────

    def _update_alpha(self, obs) -> float:
        """
        엔트로피 온도 자동 조정 (SAC v2):
            L_α = E[-α · (log π(a|s) + H̄)]
        현재 정책 엔트로피가 목표보다 낮으면 α 증가 (더 탐험)
        목표보다 높으면 α 감소 (더 집중)
        """
        with torch.no_grad():
            _, log_prob = self.actor(obs)

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy)).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        return float(alpha_loss.item())

    # ── Soft / Hard Update ─────────────────────────────

    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        """θ_target ← τ·θ + (1-τ)·θ_target"""
        for t_p, s_p in zip(target.parameters(), source.parameters()):
            t_p.data.copy_(tau * s_p.data + (1 - tau) * t_p.data)

    def _hard_update(self, target: nn.Module, source: nn.Module):
        target.load_state_dict(source.state_dict())

    # ── 저장 / 로드 ────────────────────────────────────

    def save(self, name: str = "sac_agent") -> Path:
        path = CKPT_DIR / f"{name}.pt"
        torch.save({
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt":     self.actor_opt.state_dict(),
            "critic_opt":    self.critic_opt.state_dict(),
            "log_alpha":     self.log_alpha.detach().cpu(),
            "total_updates": self._total_updates,
        }, path)
        logger.info(f"체크포인트 저장: {path}")
        return path

    def load(self, name: str = "sac_agent") -> "SACAgent":
        path = CKPT_DIR / f"{name}.pt"
        ckpt = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.log_alpha   = ckpt["log_alpha"].to(self.device).requires_grad_(self.cfg.auto_alpha)
        self._total_updates = ckpt.get("total_updates", 0)

        logger.info(f"체크포인트 로드: {path}")
        return self

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().item())
