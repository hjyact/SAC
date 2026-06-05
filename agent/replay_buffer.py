"""
agent/replay_buffer.py — 경험 리플레이 버퍼

Off-policy 학습의 핵심.
Prioritized Experience Replay (PER) 선택적 지원.

참고: Mnih et al. (2015), Schaul et al. (2015)
"""

import numpy as np
from typing import Tuple, Optional


class ReplayBuffer:
    """
    균일 샘플링 경험 리플레이 버퍼.
    메모리 효율을 위해 numpy 기반 circular buffer 사용.
    """

    def __init__(self, obs_dim: int, action_dim: int, capacity: int = 100_000):
        self.capacity   = capacity
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self._ptr       = 0
        self._size      = 0

        self.obs     = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1),          dtype=np.float32)
        self.next_obs= np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.dones   = np.zeros((capacity, 1),          dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs[self._ptr]      = obs
        self.actions[self._ptr]  = action
        self.rewards[self._ptr]  = reward
        self.next_obs[self._ptr] = next_obs
        self.dones[self._ptr]    = float(done)

        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        idx = np.random.randint(0, self._size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        return self._size >= 256
