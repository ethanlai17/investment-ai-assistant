import yfinance as yf
import pandas as pd
from loguru import logger


SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}

_FALLBACK_ETF = "SPY"


class SectorETFFetcher:
    def fetch(
        self, tickers: list[str], lookback_days: int = 365
    ) -> tuple[dict[str, str], dict[str, str], dict[str, pd.DataFrame]]:
        """
        Returns:
          ticker_sector: {ticker -> sector name}
          ticker_etf:    {ticker -> ETF symbol}
          etf_prices:    {etf_symbol -> price DataFrame}
        """
        ticker_sector: dict[str, str] = {}
        ticker_etf: dict[str, str] = {}
        etfs_needed: set[str] = set()

        for t in tickers:
            try:
                sector = yf.Ticker(t).info.get("sector", "")
                etf = SECTOR_ETF_MAP.get(sector, _FALLBACK_ETF)
                ticker_sector[t] = sector or "Unknown"
                ticker_etf[t] = etf
                etfs_needed.add(etf)
                logger.debug(f"{t}: sector={sector}, ETF={etf}")
            except Exception as exc:
                logger.warning(f"{t}: sector lookup failed — {exc}")
                ticker_sector[t] = "Unknown"
                ticker_etf[t] = _FALLBACK_ETF
                etfs_needed.add(_FALLBACK_ETF)

        etf_prices: dict[str, pd.DataFrame] = {}
        for etf in etfs_needed:
            try:
                df = yf.download(etf, period=f"{lookback_days + 10}d", auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[["Close"]].dropna()
                etf_prices[etf] = df
                logger.debug(f"{etf}: {len(df)} rows fetched")
            except Exception as exc:
                logger.warning(f"{etf}: ETF price fetch failed — {exc}")

        return ticker_sector, ticker_etf, etf_prices
