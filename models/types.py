from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


@dataclass
class RawNewsItem:
    headline: str
    summary: str
    published_at: datetime
    source: str
    url: str
    raw_tickers: list[str] = field(default_factory=list)


@dataclass
class NewsItem:
    headline: str
    summary: str
    published_at: datetime
    source: str
    url: str
    tickers: list[str]
    content_hash: str


@dataclass
class SentimentResult:
    label: str  # "positive" | "neutral" | "negative"
    confidence: float  # 0.0–1.0


@dataclass
class ScoredNewsItem:
    news_item: NewsItem
    sentiment: SentimentResult


@dataclass
class TickerSentiment:
    ticker: str
    date: str  # YYYY-MM-DD
    avg_score: float  # -1 to +1
    article_count: int
    positive_count: int
    neutral_count: int
    negative_count: int
    top_headlines: list[str]  # up to 3


@dataclass
class AnalystData:
    ticker: str
    consensus: str  # "Strong Buy"|"Buy"|"Hold"|"Sell"|"Strong Sell"|"Unknown"
    price_target: float  # mean analyst price target; 0.0 if unavailable
    analyst_count: int
    source: str  # "tipranks" | "yfinance"


@dataclass
class Prediction:
    ticker: str
    up_probability: float  # 0.0–1.0
    confidence: float
    feature_snapshot: dict


class Signal(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    AVOID = "AVOID"


@dataclass
class Recommendation:
    ticker: str
    signal: Signal
    combined_score: float
    sentiment_score: float  # normalised 0–1
    ml_up_probability: float
    analyst_data: AnalystData | None
    analyst_score: float  # normalised 0–1; 0.0 if no data
    confidence: float
    key_news: list[str]
    ticker_insight: str  # Claude Haiku one-liner
    current_price: float
    price_change_pct: float


@dataclass
class ReportData:
    date: str
    market_summary: str
    recommendations: list[Recommendation]
    ticker_sentiments: dict[str, TickerSentiment]
    ticker_predictions: dict[str, Prediction]
    top_picks: list[Recommendation]
    notes: str
