# SAC 트레이딩 시스템

Soft Actor-Critic 기반 연속 행동 공간 자동 트레이딩 에이전트

---

## 적용 이론

| 이론 | 출처 | 적용 위치 |
|------|------|-----------|
| **Soft Actor-Critic** | Haarnoja et al. (2018, 2019) | `agent/sac_agent.py` |
| **Twin Critics** | Fujimoto et al. (TD3, 2018) | `networks/sac_nets.py` |
| **Squashed Gaussian** | SAC 논문 | `networks/sac_nets.py` |
| **Auto-entropy tuning** | SAC v2 | `agent/sac_agent.py` |
| **Sharpe/Sortino Reward** | Sharpe(1966), Sortino(1991) | `env/trading_env.py` |
| **Kelly Criterion** | Kelly (1956) | `env/trading_env.py`, `evaluation/` |
| **Hurst Exponent** | Hurst (1951), R/S Analysis | `utils/features.py` |
| **Garman-Klass Vol** | Garman & Klass (1980) | `utils/features.py` |
| **Amihud Illiquidity** | Amihud (2002) | `utils/features.py` |
| **Roll Spread** | Roll (1984) | `utils/features.py` |
| **Layer Normalization** | Ba et al. (2016) | `networks/sac_nets.py` |

---

## 프로젝트 구조

```
sac_trader/
├── config.py                  # 전역 설정 (모든 하이퍼파라미터)
├── main.py                    # 진입점
│
├── utils/
│   └── features.py            # 이론 기반 피처 엔지니어링
│
├── env/
│   └── trading_env.py         # Gymnasium 트레이딩 환경
│                              #  - 연속 행동: 목표 포지션 [-1, 1]
│                              #  - 4가지 보상 설계
│
├── networks/
│   └── sac_nets.py            # Actor (Squashed Gaussian) + Twin Critic
│
├── agent/
│   ├── sac_agent.py           # SAC 알고리즘 구현
│   └── replay_buffer.py       # 경험 리플레이 버퍼
│
├── training/
│   └── trainer.py             # 학습 루프 (워밍업, 평가, 저장)
│
└── evaluation/
    └── evaluator.py           # 백테스트 성과 분석 + 시각화
```

---

## 설치 및 실행

```bash
pip install torch gymnasium numpy pandas scikit-learn matplotlib yfinance pyarrow

# 빠른 테스트 (1000 스텝, 합성 데이터)
python main.py --mode test

# 실제 학습 (합성 데이터, 200,000 스텝)
python main.py --mode train --use-synthetic --steps 200000

# 실제 종목 학습 (네트워크 필요)
python main.py --mode train --ticker AAPL --steps 500000

# 보상 함수 변경
python main.py --mode train --reward sharpe
python main.py --mode train --reward mixed

# 암호화폐 학습 (ccxt 필요: pip install ccxt)
python main.py --mode train --crypto                                    # BTC/USDT 일봉 (기본)
python main.py --mode train --crypto --symbol ETH/USDT --timeframe 1h  # ETH 1시간봉
python main.py --mode train --crypto --symbol BTC/USDT --timeframe 4h --steps 500000
python main.py --mode train --crypto --exchange bybit --symbol BTC/USDT

# 저장된 모델 평가
python main.py --mode eval --load best_sac
```

---

## SAC 핵심 수식

### ① Critic 손실 (Soft Bellman Backup)
```
y = r + γ(1-d) · [min_Q(s', ã') - α·log π(ã'|s')]
L_Q = E[(Q(s,a) - y)²]
```

### ② Actor 손실 (정책 개선)
```
L_π = E[α·log π(a|s) - min_Q(s,a)]
```

### ③ 엔트로피 온도 자동 조정 (SAC v2)
```
L_α = E[-α · (log π(a|s) + H̄)]
H̄ = -action_dim  (목표 엔트로피)
```

### ④ Soft Target Update
```
θ_target ← τ·θ + (1-τ)·θ_target   (τ = 0.005)
```

---

## 보상 함수 비교

| 보상 타입 | 수식 | 특징 |
|-----------|------|------|
| `pnl` | step_return | 단순, 불안정 |
| `sharpe` | rolling Sharpe (30 step) | 위험 조정, 안정적 |
| `sortino` | rolling Sortino | 하방 위험만 패널티 |
| `mixed` | Sharpe + MDD패널티 + 거래비용 + Kelly패널티 | 가장 현실적 |

---

## 주요 설정 (`config.py`)

```python
# 환경
env_cfg.reward_type    = "mixed"   # 보상 함수
env_cfg.window_size    = 30        # 관측 윈도우
env_cfg.episode_length = 252       # 에피소드 길이 (거래일)

# SAC
sac_cfg.hidden_dims    = [256, 256]  # 네트워크 크기
sac_cfg.gamma          = 0.99        # 할인율
sac_cfg.tau            = 0.005       # Soft update 계수
sac_cfg.auto_alpha     = True        # α 자동 조정

# 학습
train_cfg.total_timesteps = 200_000
train_cfg.eval_interval   = 5_000
```

---

## 평가 지표

| 지표 | 설명 |
|------|------|
| Sharpe Ratio | (수익 - 무위험) / 변동성, >1 우수 |
| Sortino Ratio | 하방 위험 기준 Sharpe |
| Calmar Ratio | CAGR / \|MDD\| |
| Kelly Fraction | 이론적 최적 배팅 비율 |
| Profit Factor | 총이익 / 총손실, >1.5 우수 |
