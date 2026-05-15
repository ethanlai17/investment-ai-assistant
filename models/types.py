from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import pandas as pd


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
    top_news_urls: list[str]  # parallel list of URLs


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
class FundamentalData:
    ticker: str
    pe_ratio: float | None
    pb_ratio: float | None
    roe: float | None
    roa: float | None
    earnings_growth: float | None
    revenue_growth: float | None
    fcf_yield: float | None
    debt_to_equity: float | None
    current_ratio: float | None
    operating_margin: float | None
    market_cap: float | None


@dataclass
class MacroData:
    spy_returns: pd.Series
    vix_levels: pd.Series
    yield_spread: pd.Series
    current_vix: float
    current_spread: float


@dataclass
class RegimeState:
    current_state: str  # "bull" | "bear"
    regime_score: float  # [0, 1]
    bull_probability: float
    vix_signal: float
    yield_curve_signal: float


@dataclass
class RelativeStrength:
    ticker: str
    rs_52w: float
    rs_26w: float
    rs_13w: float
    rs_score: float  # [0, 1]
    sector: str
    sector_etf: str


@dataclass
class FactorExposure:
    ticker: str
    alpha: float          # annualised raw alpha
    beta_mkt: float
    beta_smb: float
    beta_hml: float
    beta_umd: float
    r_squared: float
    carhart_alpha: float  # cross-sectional rank [0, 1]


@dataclass
class RiskMetrics:
    ticker: str
    sharpe_ratio: float
    max_drawdown: float
    beta: float
    annualised_vol: float
    cvar_95: float        # daily CVaR at 95% (negative)
    sortino_ratio: float
    risk_score: float     # [0, 1]
    is_high_risk: bool


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
    key_news_urls: list[str]
    ticker_insight: str  # populated later by ReportGenerator
    current_price: float
    price_change_pct: float
    fundamental_score: float = 0.0
    regime_score: float = 0.0
    rs_score: float = 0.0
    risk_score: float = 0.0
    risk_metrics: RiskMetrics | None = None
    carhart_alpha: float = 0.0
    pt_upside: float = 0.0


@dataclass
class ReportData:
    date: str
    market_summary: str
    recommendations: list[Recommendation]
    ticker_sentiments: dict[str, TickerSentiment]
    ticker_predictions: dict[str, Prediction]
    top_picks: list[Recommendation]
    notes: str
