import json

import requests
import yfinance as yf
from bs4 import BeautifulSoup
from loguru import logger

from models.types import AnalystData

_TIPRANKS_URL = "https://www.tipranks.com/stocks/{ticker}/forecast"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Consensus label mapping from TipRanks numeric scores
_TIPRANKS_CONSENSUS_MAP = {
    "Strong Buy": "Strong Buy",
    "Moderate Buy": "Buy",
    "Buy": "Buy",
    "Hold": "Hold",
    "Neutral": "Hold",
    "Moderate Sell": "Sell",
    "Sell": "Sell",
    "Strong Sell": "Strong Sell",
}

_YFINANCE_CONSENSUS_THRESHOLDS = [
    (0.6, "Strong Buy"),
    (0.45, "Buy"),
    (0.3, "Hold"),
    (0.15, "Sell"),
    (0.0, "Strong Sell"),
]


def _yfinance_consensus(strong_buy: int, buy: int, hold: int, sell: int, strong_sell: int) -> str:
    total = strong_buy + buy + hold + sell + strong_sell
    if total == 0:
        return "Unknown"
    score = (strong_buy * 1.0 + buy * 0.75 + hold * 0.5 + sell * 0.25 + strong_sell * 0.0) / total
    for threshold, label in _YFINANCE_CONSENSUS_THRESHOLDS:
        if score >= threshold:
            return label
    return "Unknown"


class AnalystFetcher:
    def fetch(self, tickers: list[str]) -> dict[str, AnalystData]:
        result: dict[str, AnalystData] = {}
        for ticker in tickers:
            data = self._fetch_tipranks(ticker)
            if data is None:
                logger.debug(f"{ticker}: TipRanks unavailable, using yfinance analyst data")
                data = self._fetch_yfinance_fallback(ticker)
            result[ticker] = data
            logger.debug(
                f"{ticker}: analyst consensus={data.consensus}, "
                f"target=${data.price_target:.2f}, source={data.source}"
            )
        return result

    def _fetch_tipranks(self, ticker: str) -> AnalystData | None:
        try:
            url = _TIPRANKS_URL.format(ticker=ticker.lower())
            resp = requests.get(url, headers=_HEADERS, timeout=10)
            if resp.status_code != 200:
                return None

            soup = BeautifulSoup(resp.text, "html.parser")
            script_tag = soup.find("script", id="__NEXT_DATA__")
            if not script_tag:
                return None

            data = json.loads(script_tag.string)
            # Navigate TipRanks Next.js page props to find analyst consensus
            props = (
                data.get("props", {})
                .get("pageProps", {})
                .get("data", {})
            )
            consensus_data = props.get("consensusOverview", {})
            consensus_raw = consensus_data.get("consensus", "")
            consensus = _TIPRANKS_CONSENSUS_MAP.get(consensus_raw, "Unknown")
            price_target = float(consensus_data.get("priceTarget", 0.0) or 0.0)
            analyst_count = int(consensus_data.get("numOfAnalysts", 0) or 0)

            if consensus == "Unknown" and price_target == 0.0:
                return None

            return AnalystData(
                ticker=ticker,
                consensus=consensus,
                price_target=price_target,
                analyst_count=analyst_count,
                source="tipranks",
            )
        except Exception as exc:
            logger.debug(f"{ticker}: TipRanks parse error — {exc}")
            return None

    def _fetch_yfinance_fallback(self, ticker: str) -> AnalystData:
        try:
            t = yf.Ticker(ticker)

            price_target = 0.0
            try:
                targets = t.analyst_price_targets
                if targets and isinstance(targets, dict):
                    price_target = float(targets.get("mean", 0.0) or 0.0)
            except Exception:
                pass

            strong_buy = buy = hold = sell = strong_sell = 0
            try:
                summary = t.recommendations_summary
                if summary is not None and not summary.empty:
                    row = summary.iloc[0]
                    strong_buy = int(row.get("strongBuy", 0) or 0)
                    buy = int(row.get("buy", 0) or 0)
                    hold = int(row.get("hold", 0) or 0)
                    sell = int(row.get("sell", 0) or 0)
                    strong_sell = int(row.get("strongSell", 0) or 0)
            except Exception:
                pass

            consensus = _yfinance_consensus(strong_buy, buy, hold, sell, strong_sell)
            analyst_count = strong_buy + buy + hold + sell + strong_sell

            return AnalystData(
                ticker=ticker,
                consensus=consensus,
                price_target=price_target,
                analyst_count=analyst_count,
                source="yfinance",
            )
        except Exception as exc:
            logger.warning(f"{ticker}: yfinance analyst fallback failed — {exc}")
            return AnalystData(
                ticker=ticker,
                consensus="Unknown",
                price_target=0.0,
                analyst_count=0,
                source="yfinance",
            )
