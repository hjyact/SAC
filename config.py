"""
config.py — SAC 트레이딩 시스템 전역 설정

참고 이론:
  - SAC (Haarnoja et al., 2018): 엔트로피 최대화 + Off-policy Actor-Critic
  - Kelly Criterion: 최적 포지션 사이징
  - Markowitz MPT: 리스크-수익 트레이드오프
  - Temporal Difference: TD(λ) 기반 가치 추정
"""

from dataclasses import dataclass, field
from pathlib import Path
import torch

ROOT  = Path(__file__).parent
CKPT_DIR   = ROOT / "checkpoints";  CKPT_DIR.mkdir(exist_ok=True)
LOG_DIR    = ROOT / "logs";         LOG_DIR.mkdir(exist_ok=True)
RESULT_DIR = ROOT / "results";      RESULT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── 환경 설정 ──────────────────────────────────────────
@dataclass
class EnvConfig:
    # 관측 윈도우 (과거 몇 봉을 state로 볼 것인가)
    window_size: int        = 30

    # 포트폴리오
    initial_capital: float  = 1_000_000.0   # 초기 자본 (원)
    commission: float       = 0.00015        # 편도 수수료
    slippage: float         = 0.0001         # 슬리피지
    max_position: float     = 1.0            # 최대 포지션 (자본의 100%)

    # 보상 설계
    reward_type: str        = "sharpe"       # "pnl" | "sharpe" | "sortino" | "mixed"
    reward_scaling: float   = 1.0
    risk_penalty: float     = 0.1            # 과도한 리스크 패널티 계수
    drawdown_penalty: float = 0.5            # MDD 패널티

    # 에피소드
    episode_length: int     = 252            # 1 에피소드 = 252 거래일 (1년)
    use_random_start: bool  = True           # 랜덤 시작점 (과적합 방지)


# ── SAC 하이퍼파라미터 ─────────────────────────────────
@dataclass
class SACConfig:
    # 네트워크
    hidden_dims: list       = field(default_factory=lambda: [256, 256])
    activation: str         = "relu"         # "relu" | "tanh" | "elu"

    # 학습률
    actor_lr: float         = 3e-4
    critic_lr: float        = 3e-4
    alpha_lr: float         = 3e-4           # 엔트로피 온도 자동 조정

    # SAC 핵심 파라미터
    gamma: float            = 0.99           # 할인율
    tau: float              = 0.005          # Soft target update 계수
    alpha: float            = 0.2            # 초기 엔트로피 온도
    auto_alpha: bool        = True           # 자동 엔트로피 조정 (SAC-v2)
    target_entropy: float   = -1.0           # 목표 엔트로피 (-action_dim)

    # 리플레이 버퍼
    buffer_size: int        = 100_000
    batch_size: int         = 256
    min_replay_size: int    = 1_000          # 학습 시작 전 워밍업

    # 학습
    gradient_steps: int     = 1             # 환경 스텝당 gradient 업데이트 횟수
    target_update_interval: int = 1         # target network 업데이트 주기
    grad_clip: float        = 1.0           # gradient clipping


# ── 피처 설정 ──────────────────────────────────────────
@dataclass
class FeatureConfig:
    # 기술적 지표 파라미터
    rsi_period: int         = 14
    macd_fast: int          = 12
    macd_slow: int          = 26
    macd_signal: int        = 9
    bb_period: int          = 20
    atr_period: int         = 14
    vol_windows: list       = field(default_factory=lambda: [5, 10, 20])
    ma_windows: list        = field(default_factory=lambda: [5, 10, 20, 60])

    # 추가 이론 기반 피처
    use_hurst: bool         = True    # Hurst 지수 (장기기억 / 평균회귀 판별)
    use_microstructure: bool= True    # 시장 미시구조 (bid-ask spread proxy)
    use_regime: bool        = True    # HMM 기반 레짐 피처 (변동성 레짐)


# ── 학습 설정 ──────────────────────────────────────────
@dataclass
class TrainConfig:
    total_timesteps: int    = 200_000
    eval_interval: int      = 5_000
    eval_episodes: int      = 3
    save_interval: int      = 10_000
    log_interval: int       = 1_000
    seed: int               = 42


env_cfg     = EnvConfig()
sac_cfg     = SACConfig()
feat_cfg    = FeatureConfig()
train_cfg   = TrainConfig()
