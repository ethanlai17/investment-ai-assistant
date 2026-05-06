import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands


_FEATURE_COLS = [
    "rsi_14", "macd_line", "macd_signal", "macd_hist",
    "bb_pct", "volume_ratio", "mom_5", "mom_20",
    "daily_return", "sentiment_score",
    "fundamental_score", "regime_score", "rs_score", "risk_score",
]


def compute_features(
    df: pd.DataFrame,
    sentiment_score: float,
    fundamental_score: float = 0.5,
    regime_score: float = 0.5,
    rs_score: float = 0.5,
    risk_score: float = 0.5,
) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    rsi = RSIIndicator(close=close, window=14)
    df["rsi_14"] = rsi.rsi()

    macd = MACD(close=close)
    df["macd_line"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    bb = BollingerBands(close=close, window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    bb_range = bb_upper - bb_lower
    df["bb_pct"] = np.where(bb_range != 0, (close - bb_lower) / bb_range, 0.5)

    vol_ma20 = volume.rolling(20).mean()
    df["volume_ratio"] = np.where(vol_ma20 != 0, volume / vol_ma20, 1.0)

    df["mom_5"] = close.pct_change(5)
    df["mom_20"] = close.pct_change(20)
    df["daily_return"] = close.pct_change(1)
    df["sentiment_score"] = sentiment_score
    df["fundamental_score"] = fundamental_score
    df["regime_score"] = regime_score
    df["rs_score"] = rs_score
    df["risk_score"] = risk_score

    return df


def build_training_set(
    df: pd.DataFrame, window_days: int = 252, forward_days: int = 20
) -> tuple[pd.DataFrame, pd.Series]:
    df = df.tail(window_days + forward_days + 1).copy()
    close = df["Close"].squeeze()
    df["target_up"] = (close.shift(-forward_days) > close).astype(int)
    df = df.dropna(subset=_FEATURE_COLS + ["target_up"])
    # Drop last forward_days rows (no valid target)
    df = df.iloc[:-forward_days] if len(df) > forward_days else df
    X = df[_FEATURE_COLS]
    y = df["target_up"]
    return X, y


def build_prediction_row(
    df: pd.DataFrame,
    sentiment_score: float,
    fundamental_score: float = 0.5,
    regime_score: float = 0.5,
    rs_score: float = 0.5,
    risk_score: float = 0.5,
) -> pd.DataFrame:
    enriched = compute_features(
        df, sentiment_score, fundamental_score, regime_score, rs_score, risk_score
    )
    latest = enriched[_FEATURE_COLS].dropna().tail(1)
    return latest
