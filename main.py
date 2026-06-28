"""
main.py — SAC 트레이딩 시스템 진입점

사용법:
    python main.py                              # 기본 실행 (합성 데이터)
    python main.py --ticker AAPL --mode train   # 실제 데이터 학습
    python main.py --mode eval --load best_sac  # 저장 모델 평가
    python main.py --reward mixed --steps 500000

참고 이론 요약:
  ┌─────────────────────────────────────────────────────────┐
  │ SAC (Haarnoja 2018)    : 엔트로피 최대화 Off-policy RL  │
  │ Twin Critics (TD3)     : Q값 과대추정 방지              │
  │ Squashed Gaussian      : 연속 행동 경계 처리            │
  │ Auto-α (SAC v2)        : 탐험-활용 자동 균형            │
  │ Sharpe 보상            : 위험 조정 수익 최적화          │
  │ Kelly Criterion        : 포지션 사이징 이론             │
  │ Hurst Exponent         : 시장 레짐 감지                 │
  │ Garman-Klass Vol       : OHLC 기반 효율적 변동성 추정   │
  │ Amihud Illiquidity     : 유동성 리스크 피처             │
  │ Roll Spread            : 시장 미시구조 피처             │
  └─────────────────────────────────────────────────────────┘
"""

import argparse
import logging
import numpy as np
import pandas as pd
import torch
import sys

from config import (
    env_cfg, sac_cfg, feat_cfg, train_cfg, DEVICE,
    EnvConfig, SACConfig, FeatureConfig, TrainConfig,
)
from utils.features import build_all_features
from env.trading_env import TradingEnv
from agent.sac_agent import SACAgent
from agent.replay_buffer import ReplayBuffer
from training.trainer import SACTrainer
from evaluation.evaluator import run_evaluation, plot_training_curves, plot_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── 데이터 준비 ────────────────────────────────────────

def prepare_data(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    데이터 로드 → 피처 생성 → 훈련/테스트 분리.
    시간 순서 유지, look-ahead bias 없음.
    """
    if args.use_synthetic or args.ticker is None:
        logger.info("합성 데이터 생성 중...")
        price_df = _make_synthetic_data(n=2000, seed=42)
    else:
        try:
            import yfinance as yf
            logger.info(f"데이터 다운로드: {args.ticker}")
            raw = yf.download(args.ticker, start=args.start, end=args.end,
                              interval=args.interval, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            price_df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if price_df.empty:
                raise ValueError("downloaded data is empty")
        except Exception as e:
            logger.warning(f"다운로드 실패 ({e}), 합성 데이터 사용")
            price_df = _make_synthetic_data(n=2000, seed=42)

    logger.info(f"원본 데이터: {len(price_df)}행")

    # 피처 엔지니어링
    logger.info("피처 엔지니어링 중...")
    feat_df = build_all_features(price_df, feat_cfg)

    # 유효 행 필터링
    common = feat_df.index.intersection(price_df.index)
    feat_df  = feat_df.loc[common].dropna()
    price_df = price_df.loc[feat_df.index]

    logger.info(f"유효 데이터: {len(feat_df)}행, 피처: {feat_df.shape[1]}개")

    # 훈련/테스트 분리 (시간 순서 유지)
    split = int(len(feat_df) * (1 - args.test_ratio))
    train_feat  = feat_df.iloc[:split]
    train_price = price_df.iloc[:split]
    test_feat   = feat_df.iloc[split:]
    test_price  = price_df.iloc[split:]

    logger.info(f"훈련: {len(train_feat)}행 | 테스트: {len(test_feat)}행")

    return (train_feat, train_price), (test_feat, test_price)


def _make_synthetic_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    GBM + 레짐 전환 합성 데이터.
    실제 주가의 변동성 클러스터링 및 추세 변화를 모방.
    """
    np.random.seed(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")

    # GARCH-like 변동성 클러스터링
    vol = np.zeros(n)
    vol[0] = 0.01
    for i in range(1, n):
        shock = np.random.randn()
        vol[i] = np.sqrt(0.00001 + 0.09 * (vol[i-1]*shock)**2 + 0.90 * vol[i-1]**2)

    ret = np.random.randn(n) * vol
    # 추세 레짐 (bull/bear)
    regime = np.sin(np.linspace(0, 4*np.pi, n)) * 0.0002
    ret   += regime

    close = 100 * np.exp(ret.cumsum())
    hi_lo_spread = np.abs(np.random.randn(n)) * vol * close
    open_  = close * np.exp(-ret * np.random.uniform(0.3, 0.7, n))
    high   = close + hi_lo_spread * 0.5
    low    = close - hi_lo_spread * 0.5
    volume = np.random.lognormal(15, 1, n)

    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }, index=dates)


# ── 환경 및 에이전트 구축 ──────────────────────────────

def build_env_and_agent(
    train_data: tuple,
    test_data: tuple,
    args,
) -> tuple[TradingEnv, TradingEnv, SACAgent, ReplayBuffer]:

    train_feat, train_price = train_data
    test_feat,  test_price  = test_data

    # 환경 설정 적용
    env_cfg.reward_type     = args.reward
    env_cfg.episode_length  = args.episode_length
    env_cfg.window_size     = args.window_size

    train_env = TradingEnv(train_feat, train_price, env_cfg, mode="train")
    eval_env  = TradingEnv(test_feat,  test_price,  env_cfg, mode="eval",
                           norm_stats=train_env.norm_stats)

    obs_dim    = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]
    logger.info(f"관측 차원: {obs_dim} | 행동 차원: {action_dim} | Device: {DEVICE}")

    # SAC 설정 적용
    sac_cfg.hidden_dims = [args.hidden_size] * args.n_layers

    agent  = SACAgent(obs_dim, action_dim, sac_cfg, DEVICE)
    buffer = ReplayBuffer(obs_dim, action_dim, sac_cfg.buffer_size)

    return train_env, eval_env, agent, buffer


# ── 학습 ──────────────────────────────────────────────

def run_training(args):
    train_data, test_data = prepare_data(args)
    train_env, eval_env, agent, buffer = build_env_and_agent(train_data, test_data, args)

    train_cfg.total_timesteps = args.steps
    trainer = SACTrainer(train_env, eval_env, agent, buffer, train_cfg)

    logger.info(f"\n{'='*60}")
    logger.info("SAC 학습 시작")
    logger.info(f"{'='*60}")

    eval_history = trainer.train()
    trainer.save_results()

    if not args.no_plot and eval_history:
        plot_training_curves(eval_history, save=True)
        logger.info("학습 곡선 저장 완료")

    # 최고 모델 로드 후 최종 평가
    logger.info("\n최고 모델 로드 후 최종 평가...")
    try:
        agent.load("best_sac")
    except FileNotFoundError:
        logger.info("(저장된 최고 모델 없음, 현재 모델로 평가)")

    result = run_evaluation(agent, eval_env, n_episodes=1, name="SAC_Best")

    if not args.no_plot:
        plot_backtest(result, name="SAC_Best", save=True)

    return result


# ── 평가만 실행 ────────────────────────────────────────

def run_eval_only(args):
    train_data, test_data = prepare_data(args)
    train_env, eval_env, agent, buffer = build_env_and_agent(train_data, test_data, args)

    logger.info(f"모델 로드: {args.load}")
    agent.load(args.load)

    result = run_evaluation(agent, eval_env, n_episodes=1, name=args.load)

    if not args.no_plot:
        plot_backtest(result, name=args.load, save=True)

    return result


# ── 빠른 테스트 ────────────────────────────────────────

def run_quick_test(args):
    """설치 확인 및 단기 동작 테스트 (1000 스텝)."""
    logger.info("빠른 테스트 모드 (1000 스텝)")
    args.steps         = 1000
    args.use_synthetic = True
    train_cfg.eval_interval  = 500
    train_cfg.log_interval   = 200
    train_cfg.save_interval  = 1000
    sac_cfg.min_replay_size  = 100
    sac_cfg.buffer_size      = 5000
    sac_cfg.batch_size       = 64
    return run_training(args)


# ── CLI ───────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SAC 트레이딩 시스템")
    p.add_argument("--mode",      choices=["train","eval","test","tune","walkforward","ensemble","signal"],
                   default="test")
    p.add_argument("--ticker",    default=None,        help="종목 (None=합성 데이터)")
    p.add_argument("--start",     default="2018-01-01")
    p.add_argument("--end",       default="2024-12-31")
    p.add_argument("--interval",  default="1d")
    p.add_argument("--steps",     type=int, default=200_000)
    p.add_argument("--reward",    choices=["pnl","sharpe","sortino","mixed"], default="mixed")
    p.add_argument("--episode-length", type=int, default=252, dest="episode_length")
    p.add_argument("--window-size",    type=int, default=30,  dest="window_size")
    p.add_argument("--hidden-size",    type=int, default=256, dest="hidden_size")
    p.add_argument("--n-layers",       type=int, default=2,   dest="n_layers")
    p.add_argument("--test-ratio",     type=float, default=0.2, dest="test_ratio")
    p.add_argument("--load",      default="best_sac",  help="로드할 체크포인트 이름")
    p.add_argument("--use-synthetic",  action="store_true", dest="use_synthetic")
    p.add_argument("--no-plot",        action="store_true", dest="no_plot")
    # 튜닝 옵션
    p.add_argument("--trials",    type=int, default=30,  help="Optuna trial 수")
    p.add_argument("--timeout",   type=int, default=3600,help="튜닝 최대 시간(초)")
    # 앙상블 옵션
    p.add_argument("--n-agents",  type=int, default=3,   dest="n_agents", help="앙상블 에이전트 수")
    # Walk-forward 옵션
    p.add_argument("--wf-splits", type=int, default=5,   dest="wf_splits")
    return p.parse_args()


def run_walkforward(args):
    """Walk-Forward 재학습 실행."""
    from training.walk_forward import WalkForwardTrainer
    train_data, test_data = prepare_data(args)
    train_feat, train_price = train_data

    wf = WalkForwardTrainer(
        n_splits=args.wf_splits,
        n_steps_per_fold=args.steps // args.wf_splits,
        retrain_from_scratch=False,
    )
    summary = wf.run(train_feat, train_price, env_cfg, sac_cfg)
    return summary


def run_ensemble_mode(args):
    """앙상블 학습 실행."""
    from agent.ensemble import SACEnsemble
    train_data, test_data = prepare_data(args)
    train_feat, train_price = train_data
    test_feat, test_price   = test_data

    train_env = TradingEnv(train_feat, train_price, env_cfg, "train")
    eval_env  = TradingEnv(test_feat,  test_price,  env_cfg, "eval")

    obs_dim = train_env.observation_space.shape[0]
    act_dim = train_env.action_space.shape[0]

    ensemble = SACEnsemble(obs_dim, act_dim, n_agents=args.n_agents)
    sharpes  = ensemble.train_all(
        train_env, eval_env,
        n_steps=args.steps // args.n_agents,
    )
    logger.info(f"\n앙상블 완료 | 에이전트별 Sharpe: {[f'{s:.3f}' for s in sharpes]}")

    # 앙상블 평가
    from evaluation.evaluator import run_evaluation as _run_eval
    step_rets = []
    obs, _ = eval_env.reset()
    done = False
    while not done:
        action = ensemble.select_action(obs, method="confidence")
        obs, _, term, trunc, info = eval_env.step(action)
        step_rets.append(info["step_ret"])
        done = term or trunc

    rets   = np.array(step_rets)
    sharpe = (rets.mean() / (rets.std()+1e-8)) * np.sqrt(252)
    logger.info(f"앙상블 최종 Sharpe: {sharpe:.4f}")
    return {"sharpe": sharpe}


def run_signal_mode(args):
    """신호 생성기 데모 실행."""
    from utils.signal_generator import SignalGenerator, extract_feature_stats
    from utils.risk_manager import RiskManager, RiskConfig

    train_data, test_data = prepare_data(args)
    train_feat, train_price = train_data
    test_feat, test_price   = test_data

    # 모델 로드
    _, _, agent, _ = build_env_and_agent(train_data, test_data, args)
    try:
        agent.load(args.load)
        logger.info(f"모델 로드: {args.load}")
    except FileNotFoundError:
        logger.warning("저장된 모델 없음, 랜덤 가중치로 데모")

    feat_stats = extract_feature_stats(train_feat)
    generator  = SignalGenerator(agent, env_cfg, feat_cfg, feat_stats)
    risk_mgr   = RiskManager(RiskConfig())

    logger.info("\n=== 신호 생성 데모 (최근 10봉) ===")
    capital = env_cfg.initial_capital
    for i in range(min(10, len(test_price) - env_cfg.window_size)):
        idx     = env_cfg.window_size + i
        recent  = test_price.iloc[max(0, idx-env_cfg.window_size-30):idx]
        signal  = generator.generate(recent, current_capital=capital)
        risk    = risk_mgr.adjust(signal["position"], capital, recent)

        logger.info(
            f"[{test_price.index[idx].date()}] "
            f"신호={signal['signal']:4s} | "
            f"원시={signal['raw_action']:+.3f} | "
            f"리스크조정={risk['adjusted_position']:+.3f} | "
            f"신뢰도={signal['confidence']:.3f} | "
            f"레짐={risk['regime']}"
        )


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "test":
        run_quick_test(args)
    elif args.mode == "train":
        run_training(args)
    elif args.mode == "eval":
        run_eval_only(args)
    elif args.mode == "tune":
        from training.tuner import run_tuning
        run_tuning(args.trials, args.timeout)
    elif args.mode == "walkforward":
        run_walkforward(args)
    elif args.mode == "ensemble":
        run_ensemble_mode(args)
    elif args.mode == "signal":
        run_signal_mode(args)
