import yfinance as yf
from loguru import logger

from models.types import FundamentalData


class FundamentalFetcher:
    def fetch(self, tickers: list[str]) -> dict[str, FundamentalData | None]:
        results: dict[str, FundamentalData | None] = {}
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                market_cap = info.get("marketCap") or None
                free_cashflow = info.get("freeCashflow") or None
                fcf_yield = None
                if free_cashflow and market_cap and market_cap > 0:
                    fcf_yield = free_cashflow / market_cap
                results[ticker] = FundamentalData(
                    ticker=ticker,
                    pe_ratio=info.get("trailingPE") or None,
                    pb_ratio=info.get("priceToBook") or None,
                    roe=info.get("returnOnEquity") or None,
                    roa=info.get("returnOnAssets") or None,
                    earnings_growth=info.get("earningsGrowth") or None,
                    revenue_growth=info.get("revenueGrowth") or None,
                    fcf_yield=fcf_yield,
                    debt_to_equity=info.get("debtToEquity") or None,
                    current_ratio=info.get("currentRatio") or None,
                    operating_margin=info.get("operatingMargins") or None,
                    market_cap=market_cap,
                )
                logger.debug(f"{ticker}: fundamental data fetched")
            except Exception as exc:
                logger.warning(f"{ticker}: fundamental fetch failed — {exc}")
                results[ticker] = None
        return results
