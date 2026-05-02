from models.types import TickerSentiment, Prediction, AnalystData, Recommendation, Signal


_ANALYST_SCORE_MAP = {
    "Strong Buy": 1.0,
    "Buy": 0.75,
    "Hold": 0.5,
    "Sell": 0.25,
    "Strong Sell": 0.0,
}

_BUY_THRESHOLD = 0.65
_AVOID_THRESHOLD = 0.45
_CONFIDENCE_FLOOR = 0.60


def _normalize_sentiment(avg_score: float) -> float:
    return (avg_score + 1.0) / 2.0


def _analyst_normalized(analyst_data: AnalystData | None) -> float | None:
    if analyst_data is None or analyst_data.consensus == "Unknown":
        return None
    return _ANALYST_SCORE_MAP.get(analyst_data.consensus)


def _compute_combined(
    norm_sentiment: float,
    ml_up_probability: float,
    analyst_norm: float | None,
) -> float:
    if analyst_norm is not None:
        return 0.30 * norm_sentiment + 0.50 * ml_up_probability + 0.20 * analyst_norm
    return 0.40 * norm_sentiment + 0.60 * ml_up_probability


def _compute_signal(combined_score: float, confidence: float) -> Signal:
    if combined_score >= _BUY_THRESHOLD and confidence >= _CONFIDENCE_FLOOR:
        return Signal.BUY
    if combined_score >= _AVOID_THRESHOLD:
        return Signal.HOLD
    return Signal.AVOID


def recommend(
    ticker: str,
    sentiment: TickerSentiment,
    prediction: Prediction,
    analyst_data: AnalystData | None,
    current_price: float,
    price_change_pct: float,
) -> Recommendation:
    norm_sentiment = _normalize_sentiment(sentiment.avg_score)
    analyst_norm = _analyst_normalized(analyst_data)
    analyst_score = analyst_norm if analyst_norm is not None else 0.0
    combined_score = _compute_combined(norm_sentiment, prediction.up_probability, analyst_norm)
    signal = _compute_signal(combined_score, prediction.confidence)

    return Recommendation(
        ticker=ticker,
        signal=signal,
        combined_score=round(combined_score, 4),
        sentiment_score=round(norm_sentiment, 4),
        ml_up_probability=round(prediction.up_probability, 4),
        analyst_data=analyst_data,
        analyst_score=round(analyst_score, 4),
        confidence=round(prediction.confidence, 4),
        key_news=sentiment.top_headlines,
        ticker_insight="",  # populated later by ReportGenerator
        current_price=round(current_price, 2),
        price_change_pct=round(price_change_pct, 2),
    )


def rank_recommendations(recommendations: list[Recommendation]) -> list[Recommendation]:
    signal_order = {Signal.BUY: 0, Signal.HOLD: 1, Signal.AVOID: 2}
    return sorted(
        recommendations,
        key=lambda r: (signal_order[r.signal], -r.combined_score),
    )
