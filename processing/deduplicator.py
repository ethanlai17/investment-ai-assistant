import hashlib
from difflib import SequenceMatcher

from models.types import RawNewsItem, NewsItem
from processing.text_cleaner import clean_news_item


def content_hash(headline: str, summary: str) -> str:
    combined = (headline + summary).lower().strip()
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def deduplicate(items: list[RawNewsItem], threshold: float = 0.85) -> list[RawNewsItem]:
    kept: list[RawNewsItem] = []
    for item in items:
        duplicate = False
        for existing in kept:
            if _similarity(item.headline, existing.headline) >= threshold:
                duplicate = True
                # Merge ticker associations onto the surviving item
                for t in item.raw_tickers:
                    if t not in existing.raw_tickers:
                        existing.raw_tickers.append(t)
                break
        if not duplicate:
            kept.append(item)
    return kept


def map_tickers(item: RawNewsItem, tickers: list[str]) -> list[str]:
    text = (item.headline + " " + item.summary).upper()
    mentioned = [t for t in tickers if t.upper() in text]
    return mentioned if mentioned else list(item.raw_tickers)


def process(raw_items: list[RawNewsItem], tickers: list[str]) -> list[NewsItem]:
    deduped = deduplicate(raw_items)
    result: list[NewsItem] = []
    for raw in deduped:
        clean_headline, clean_summary = clean_news_item(raw.headline, raw.summary)
        if not clean_headline:
            continue
        result.append(NewsItem(
            headline=clean_headline,
            summary=clean_summary,
            published_at=raw.published_at,
            source=raw.source,
            url=raw.url,
            tickers=map_tickers(raw, tickers),
            content_hash=content_hash(clean_headline, clean_summary),
        ))
    return result
