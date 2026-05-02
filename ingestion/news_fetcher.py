from datetime import datetime, timezone, timedelta

import feedparser
import yfinance as yf
from loguru import logger

from models.types import RawNewsItem


_YAHOO_RSS = "http://finance.yahoo.com/rss/headline?s={ticker}"


def _parse_published(entry) -> datetime:
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


class NewsFetcher:
    def fetch(self, tickers: list[str], lookback_days: int) -> list[RawNewsItem]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        all_items: list[RawNewsItem] = []
        seen_urls: set[str] = set()

        for ticker in tickers:
            items = self._fetch_rss(ticker, cutoff)
            if not items:
                logger.warning(f"{ticker}: RSS returned nothing, falling back to yfinance")
                items = self._fetch_yfinance(ticker, cutoff)

            new_items = [i for i in items if i.url not in seen_urls]
            seen_urls.update(i.url for i in new_items)
            all_items.extend(new_items)
            logger.debug(f"{ticker}: fetched {len(new_items)} news items")

        logger.info(f"News fetch complete — {len(all_items)} total items across {len(tickers)} tickers")
        return all_items

    def _fetch_rss(self, ticker: str, cutoff: datetime) -> list[RawNewsItem]:
        try:
            url = _YAHOO_RSS.format(ticker=ticker)
            feed = feedparser.parse(url)
            items = []
            for entry in feed.entries:
                pub = _parse_published(entry)
                if pub < cutoff:
                    continue
                items.append(RawNewsItem(
                    headline=entry.get("title", ""),
                    summary=entry.get("summary", entry.get("description", "")),
                    published_at=pub,
                    source=feed.feed.get("title", "Yahoo Finance"),
                    url=entry.get("link", ""),
                    raw_tickers=[ticker],
                ))
            return items
        except Exception as exc:
            logger.warning(f"{ticker}: RSS fetch failed — {exc}")
            return []

    def _fetch_yfinance(self, ticker: str, cutoff: datetime) -> list[RawNewsItem]:
        try:
            news = yf.Ticker(ticker).news or []
            items = []
            for article in news:
                content = article.get("content", {})
                pub_ts = content.get("pubDate") or article.get("providerPublishTime")
                if pub_ts:
                    if isinstance(pub_ts, str):
                        pub = datetime.fromisoformat(pub_ts.replace("Z", "+00:00"))
                    else:
                        pub = datetime.fromtimestamp(pub_ts, tz=timezone.utc)
                else:
                    pub = datetime.now(timezone.utc)

                if pub < cutoff:
                    continue

                title = content.get("title") or article.get("title", "")
                summary = content.get("summary") or article.get("summary", "")
                url = content.get("canonicalUrl", {}).get("url") or article.get("link", "")
                provider = content.get("provider", {}).get("displayName") or article.get("publisher", "yfinance")

                items.append(RawNewsItem(
                    headline=title,
                    summary=summary,
                    published_at=pub,
                    source=provider,
                    url=url,
                    raw_tickers=[ticker],
                ))
            return items
        except Exception as exc:
            logger.warning(f"{ticker}: yfinance news fetch failed — {exc}")
            return []
