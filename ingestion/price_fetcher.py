from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from loguru import logger


class PriceFetcher:
    def fetch(self, tickers: list[str], lookback_days: int) -> dict[str, pd.DataFrame]:
        end = date.today()
        start = end - timedelta(days=lookback_days)
        result: dict[str, pd.DataFrame] = {}

        for ticker in tickers:
            df = self._fetch_single(ticker, str(start), str(end))
            result[ticker] = df
            if df is not None:
                logger.debug(f"{ticker}: {len(df)} price rows fetched")
            else:
                logger.warning(f"{ticker}: price fetch failed")

        return result

    def _fetch_single(self, ticker: str, start: str, end: str) -> pd.DataFrame | None:
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if df.empty:
                return None
            # Flatten MultiIndex columns produced by yf.download for a single ticker
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as exc:
            logger.warning(f"{ticker}: yfinance download failed — {exc}")
            return None
