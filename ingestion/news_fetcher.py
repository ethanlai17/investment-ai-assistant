from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse

import feedparser
import yfinance as yf
from bs4 import BeautifulSoup
from loguru import logger

from models.types import RawNewsItem


_YAHOO_RSS = "http://finance.yahoo.com/rss/headline?s={ticker}"
_CONSENT_HOSTS = {"consent.yahoo.com", "consent.google.com"}


def _is_yahoo(url: str) -> bool:
    host = urlparse(url).hostname or ""
    return host == "yahoo.com" or host.endswith(".yahoo.com")


def _clean_url(url: str) -> str:
    """Strip consent-gate URLs that can never be resolved."""
    if not url:
        return ""
    host = urlparse(url).hostname or ""
    return "" if host in _CONSENT_HOSTS else url


def _best_rss_url(entry) -> str:
    """Return the best URL for an RSS entry.
    Scans the entry's HTML summary for a direct non-Yahoo publisher link first,
    falling back to the entry link (which may be a Yahoo Finance wrapper)."""
    summary = entry.get("summary", "") or ""
    if summary:
        try:
            soup = BeautifulSoup(summary, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if href.startswith("http") and not _is_yahoo(href):
                    return href
        except Exception:
            pass
    return _clean_url(entry.get("link", ""))


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

            # Only deduplicate on non-empty URLs — items with no URL are always kept
            new_items = [i for i in items if not i.url or i.url not in seen_urls]
            seen_urls.update(i.url for i in new_items if i.url)
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
                    url=_best_rss_url(entry),
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
                provider = content.get("provider", {}).get("displayName") or article.get("publisher", "yfinance")

                # Prefer a non-Yahoo URL (direct publisher link) when available
                link = article.get("link", "")
                canonical = (content.get("canonicalUrl") or {}).get("url", "")
                if link and not _is_yahoo(link):
                    url = link
                elif canonical and not _is_yahoo(canonical):
                    url = canonical
                else:
                    url = _clean_url(link or canonical)

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
