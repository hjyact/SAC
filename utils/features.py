"""
utils/features.py — 이론 기반 피처 엔지니어링

참고 이론:
  - Hurst Exponent (R/S Analysis): 시계열 장기기억 측정
      H > 0.5 → 추세 추종 (trending)
      H < 0.5 → 평균 회귀 (mean-reverting)
      H ≈ 0.5 → 랜덤 워크 (efficient market)

  - Market Microstructure (Roll, 1984):
      Bid-ask spread proxy = 2√(-Cov(ΔP_t, ΔP_{t-1}))

  - Realized Volatility (Andersen & Bollerslev, 1998):
      고빈도 수익률의 제곱합으로 변동성 추정

  - Garman-Klass Volatility (1980):
      OHLC 데이터 활용 변동성 추정 (종가만 쓸 때보다 효율적)

  - Amihud Illiquidity (2002):
      |수익률| / 거래대금 → 유동성 프록시
"""

import numpy as np
import pandas as pd
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


def build_all_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    OHLCV → 전체 피처 DataFrame 반환.
    모든 피처는 look-ahead bias 없이 현재까지의 데이터만 사용.
    """
    f = df.copy()

    f = _price_features(f)
    f = _technical_indicators(f, cfg)
    f = _volatility_features(f, cfg)
    f = _volume_features(f, cfg)
    f = _microstructure_features(f)

    if cfg.use_hurst:
        f = _hurst_features(f)

    # 원본 OHLCV 제거 (스케일 문제 방지)
    f.drop(columns=["Open", "High", "Low", "Close", "Volume"], inplace=True)

    # 무한값 처리
    f.replace([np.inf, -np.inf], np.nan, inplace=True)

    return f


# ── 가격 수익률 피처 ───────────────────────────────────

def _price_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]

    df["log_ret"]    = np.log(c / c.shift(1))
    df["log_ret_2"]  = np.log(c / c.shift(2))
    df["log_ret_5"]  = np.log(c / c.shift(5))
    df["log_ret_10"] = np.log(c / c.shift(10))
    df["log_ret_20"] = np.log(c / c.shift(20))

    # 고가-저가 범위
    df["hl_ratio"] = np.log(df["High"] / df["Low"])

    # 갭
    df["gap"] = np.log(df["Open"] / c.shift(1))

    # 종가 위치 (Low~High 내)
    hl = df["High"] - df["Low"] + 1e-9
    df["close_pos"] = (c - df["Low"]) / hl

    # 캔들 방향
    df["candle_dir"] = np.sign(c - df["Open"])

    return df


# ── 기술적 지표 ────────────────────────────────────────

def _technical_indicators(df: pd.DataFrame, cfg) -> pd.DataFrame:
    c = df["Close"]

    # RSI
    df["rsi"] = _rsi(c, cfg.rsi_period)
    df["rsi_norm"] = df["rsi"] / 100.0 - 0.5   # [-0.5, 0.5] 정규화

    # MACD
    ema_f = c.ewm(span=cfg.macd_fast,   adjust=False).mean()
    ema_s = c.ewm(span=cfg.macd_slow,   adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=cfg.macd_signal, adjust=False).mean()
    df["macd_hist_norm"] = (macd - sig) / (c + 1e-9)

    # Bollinger %B
    sma = c.rolling(cfg.bb_period).mean()
    std = c.rolling(cfg.bb_period).std()
    df["bb_pct_b"] = (c - (sma - 2*std)) / (4*std + 1e-9)
    df["bb_width"]  = 4 * std / (sma + 1e-9)

    # 이동평균 대비 위치 (정규화)
    for w in cfg.ma_windows:
        ma = c.rolling(w).mean()
        df[f"price_ma_{w}"] = (c - ma) / (ma + 1e-9)

    # Stochastic
    lo = df["Low"].rolling(14).min()
    hi = df["High"].rolling(14).max()
    df["stoch_k"] = (c - lo) / (hi - lo + 1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ADX (추세 강도)
    df["adx"] = _adx(df, 14)

    # CCI (Commodity Channel Index)
    tp  = (df["High"] + df["Low"] + c) / 3
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * mad + 1e-9)
    df["cci_norm"] = df["cci"].clip(-3, 3) / 3.0

    return df


def _rsi(series, period):
    delta   = series.diff()
    gain    = delta.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    loss    = (-delta).clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-9))


def _adx(df, period):
    """Average Directional Index (추세 강도 지표)"""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    up   = high - high.shift(1)
    down = low.shift(1) - low
    dm_p = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    dm_m = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    atr  = tr.ewm(span=period, adjust=False).mean()
    di_p = 100 * dm_p.ewm(span=period, adjust=False).mean() / (atr + 1e-9)
    di_m = 100 * dm_m.ewm(span=period, adjust=False).mean() / (atr + 1e-9)

    dx   = 100 * (di_p - di_m).abs() / (di_p + di_m + 1e-9)
    adx  = dx.ewm(span=period, adjust=False).mean()
    return adx / 100.0  # [0,1]


# ── 변동성 피처 ────────────────────────────────────────

def _volatility_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    c = df["Close"]
    ret = np.log(c / c.shift(1))

    # 롤링 실현 변동성 (연율화)
    for w in cfg.vol_windows:
        df[f"rvol_{w}"] = ret.rolling(w).std() * np.sqrt(252)

    # Garman-Klass 변동성 (OHLC 활용, 더 효율적)
    # σ²_GK = 0.5*(ln(H/L))² - (2ln2-1)*(ln(C/O))²
    df["gk_vol"] = np.sqrt(
        0.5 * np.log(df["High"] / df["Low"]).pow(2).rolling(20).mean()
        - (2*np.log(2)-1) * np.log(c / df["Open"]).pow(2).rolling(20).mean()
    ) * np.sqrt(252)

    # 변동성 레짐 (현재 변동성 / 장기 평균)
    long_vol = ret.rolling(60).std() * np.sqrt(252)
    df["vol_regime"] = df["rvol_20"] / (long_vol + 1e-9)

    # ATR 정규화
    prev_c = c.shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_c).abs(),
        (df["Low"]  - prev_c).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=cfg.atr_period, adjust=False).mean()
    df["atr_norm"] = atr / (c + 1e-9)

    return df


# ── 거래량 피처 ────────────────────────────────────────

def _volume_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    vol = df["Volume"]
    c   = df["Close"]

    for w in cfg.vol_windows:
        df[f"vol_ratio_{w}"] = vol / (vol.rolling(w).mean() + 1e-9)

    # OBV 모멘텀
    direction = np.sign(c.diff())
    obv       = (vol * direction).cumsum()
    df["obv_mom"] = obv / (obv.rolling(20).std() + 1e-9)

    # Amihud Illiquidity (2002)
    # ILLIQ = |r_t| / (Volume_t × Price_t)  → 유동성 부족 = 값 클수록 비유동적
    dollar_vol = vol * c
    df["amihud"] = np.log(
        np.abs(df["log_ret"]) / (dollar_vol + 1e-9) * 1e9 + 1e-9
    )

    # VWAP 대비 위치
    vwap = (c * vol).rolling(20).sum() / (vol.rolling(20).sum() + 1e-9)
    df["vwap_dev"] = (c - vwap) / (vwap + 1e-9)

    return df


# ── 시장 미시구조 피처 ─────────────────────────────────

def _microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Roll (1984) Spread Estimator:
        spread = 2 * sqrt(max(-Cov(ΔP_t, ΔP_{t-1}), 0))
    가격 충격과 유동성의 대리 지표로 활용.
    """
    ret = df["log_ret"]

    # 자기 공분산 (시차 1)
    cov = ret.rolling(20).apply(
        lambda x: np.cov(x[:-1], x[1:])[0, 1] if len(x) > 5 else 0,
        raw=True,
    )
    df["roll_spread"] = 2 * np.sqrt(np.maximum(-cov, 0))

    # 가격 반전 지표 (단기 평균회귀 신호)
    df["price_reversal_5"]  = -df["log_ret_5"]   # 단기 반전
    df["price_reversal_20"] = -df["log_ret_20"]  # 중기 반전

    # 시간 피처 (sin/cos 인코딩)
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        try:
            idx = pd.to_datetime(idx)
        except Exception:
            idx = pd.date_range("2000-01-01", periods=len(df), freq="B")

    df["dow_sin"]   = np.sin(2 * np.pi * idx.dayofweek / 5)
    df["dow_cos"]   = np.cos(2 * np.pi * idx.dayofweek / 5)
    df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)

    return df


# ── Hurst 지수 ─────────────────────────────────────────

def _hurst_features(df: pd.DataFrame, min_window: int = 40) -> pd.DataFrame:
    """
    R/S Analysis (Hurst, 1951):
        H = log(R/S) / log(n)

    롤링 윈도우로 시간에 따른 레짐 변화 감지.
    H > 0.5: 추세 지속 → 모멘텀 전략 유리
    H < 0.5: 평균회귀 → 역추세 전략 유리
    """
    ret = df["log_ret"].fillna(0).values
    n   = len(ret)
    hurst_vals = np.full(n, np.nan)

    window = 60
    for i in range(window, n):
        hurst_vals[i] = _compute_hurst(ret[max(0, i-window):i])

    df["hurst"] = hurst_vals
    df["hurst_centered"] = df["hurst"] - 0.5   # 0 중심 (음수=평균회귀, 양수=추세)

    return df


def _compute_hurst(ts: np.ndarray) -> float:
    """R/S 분석으로 Hurst 지수 계산."""
    if len(ts) < 20:
        return 0.5
    try:
        ts    = ts - ts.mean()
        cs    = np.cumsum(ts)
        R     = cs.max() - cs.min()
        S     = ts.std()
        if S < 1e-10:
            return 0.5
        return np.log(R / S) / np.log(len(ts))
    except Exception:
        return 0.5


# ── 포트폴리오 상태 피처 ───────────────────────────────

def compute_portfolio_features(
    position: float,
    unrealized_pnl_pct: float,
    cash_ratio: float,
    holding_steps: int,
    max_holding: int = 252,
) -> np.ndarray:
    """
    현재 포트폴리오 상태를 정규화된 피처로 변환.
    환경(env)의 step()에서 호출됩니다.
    """
    return np.array([
        position,                              # [-1, 1] 이미 정규화
        np.tanh(unrealized_pnl_pct * 10),     # 수익률 tanh 압축
        cash_ratio,                            # [0, 1]
        holding_steps / max_holding,           # 보유 기간 비율
    ], dtype=np.float32)
