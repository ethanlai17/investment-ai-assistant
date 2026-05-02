import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands


_FEATURE_COLS = [
    "rsi_14", "macd_line", "macd_signal", "macd_hist",
    "bb_pct", "volume_ratio", "mom_5", "mom_20",
    "daily_return", "sentiment_score",
]


def compute_features(df: pd.DataFrame, sentiment_score: float) -> pd.DataFrame:
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

    return df


def build_training_set(
    df: pd.DataFrame, window_days: int = 60
) -> tuple[pd.DataFrame, pd.Series]:
    df = df.tail(window_days + 1).copy()
    df["next_day_up"] = (df["Close"].squeeze().shift(-1) > df["Close"].squeeze()).astype(int)
    df = df.dropna(subset=_FEATURE_COLS + ["next_day_up"])
    # Drop the last row since next_day_up is NaN there
    df = df.iloc[:-1] if len(df) > 0 else df
    X = df[_FEATURE_COLS]
    y = df["next_day_up"]
    return X, y


def build_prediction_row(df: pd.DataFrame, sentiment_score: float) -> pd.DataFrame:
    enriched = compute_features(df, sentiment_score)
    latest = enriched[_FEATURE_COLS].dropna().tail(1)
    return latest
