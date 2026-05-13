import io
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from loguru import logger


class ReferenceDistributionBuilder:
    """
    Builds sorted reference arrays (RS ratios and Sharpe ratios) from the full
    S&P 500 universe so that both pipelines rank against the same distribution.
    RS ratios are computed vs SPY as a uniform benchmark across all 503 stocks.
    Results are cached on disk per calendar day.
    """

    _WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    _RISK_FREE_RATE = 0.04

    def build(self, cache_dir: str) -> dict[str, np.ndarray]:
        today = date.today().isoformat()
        cache_path = Path(cache_dir) / f".ref_{today}.npz"

        if cache_path.exists():
            logger.info("Reference distributions: loading from cache")
            data = np.load(cache_path)
            return {k: data[k] for k in ("rs_52w", "rs_26w", "rs_13w", "sharpes")}

        logger.info("Reference distributions: building from S&P 500 universe (~60s)")
        tickers = self._fetch_sp500_tickers()

        start = (date.today() - timedelta(days=400)).isoformat()
        raw = yf.download(
            tickers + ["SPY"], start=start, auto_adjust=True, progress=False
        )
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        spy = close["SPY"].dropna() if "SPY" in close.columns else pd.Series(dtype=float)

        rs_52w, rs_26w, rs_13w, sharpes = [], [], [], []
        for ticker in tickers:
            if ticker not in close.columns:
                continue
            prices = close[ticker].dropna()
            if len(prices) < 63:
                continue

            for days, bucket in ((252, rs_52w), (126, rs_26w), (63, rs_13w)):
                r = _rs_ratio(prices, spy, days)
                if r is not None:
                    bucket.append(r)

            rets = prices.pct_change().dropna()
            if len(rets) >= 30:
                ann_ret = float(rets.mean() * 252)
                ann_vol = float(rets.std() * np.sqrt(252))
                if ann_vol > 0:
                    sharpes.append((ann_ret - self._RISK_FREE_RATE) / ann_vol)

        result = {
            "rs_52w": np.sort(np.array(rs_52w, dtype=float)),
            "rs_26w": np.sort(np.array(rs_26w, dtype=float)),
            "rs_13w": np.sort(np.array(rs_13w, dtype=float)),
            "sharpes": np.sort(np.array(sharpes, dtype=float)),
        }
        np.savez(cache_path, **result)
        logger.info(
            f"Reference distributions built — {len(sharpes)} stocks, "
            f"rs windows: {len(rs_52w)}/{len(rs_26w)}/{len(rs_13w)}"
        )
        return result

    def _fetch_sp500_tickers(self) -> list[str]:
        resp = requests.get(
            self._WIKI_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=15
        )
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        return [str(s).replace(".", "-") for s in tables[0]["Symbol"].tolist()]


def _rs_ratio(prices: pd.Series, spy: pd.Series, days: int) -> float | None:
    if len(prices) < days + 1 or len(spy) < days + 1:
        return None
    s_start, s_end = float(prices.iloc[-(days + 1)]), float(prices.iloc[-1])
    spy_start, spy_end = float(spy.iloc[-(days + 1)]), float(spy.iloc[-1])
    if s_start == 0 or spy_start == 0:
        return None
    spy_ret = (spy_end - spy_start) / spy_start
    if spy_ret == 0:
        return None
    return ((s_end - s_start) / s_start) / spy_ret
