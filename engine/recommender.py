from models.types import (
    TickerSentiment, Prediction, AnalystData, Recommendation, Signal, RiskMetrics
)


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


def _pt_upside_normalized(analyst_data: AnalystData | None, current_price: float) -> float:
    if analyst_data is None or analyst_data.price_target <= 0 or current_price <= 0:
        return 0.5
    raw_upside = (analyst_data.price_target - current_price) / current_price
    clipped = max(-0.5, min(1.0, raw_upside))
    return round((clipped + 0.5) / 1.5, 4)


def _compute_combined(
    norm_sentiment: float,
    ml_up_probability: float,
    analyst_norm: float | None,
    fundamental_score: float,
    regime_score: float,
    rs_score: float,
    risk_score: float,
    carhart_alpha: float,
    pt_upside: float,
) -> float:
    if analyst_norm is not None:
        return (
            0.16 * fundamental_score
            + 0.18 * ml_up_probability
            + 0.12 * carhart_alpha
            + 0.12 * regime_score
            + 0.10 * rs_score
            + 0.09 * risk_score
            + 0.10 * norm_sentiment
            + 0.07 * analyst_norm
            + 0.06 * pt_upside
        )
    return (
        0.20 * fundamental_score
        + 0.22 * ml_up_probability
        + 0.15 * carhart_alpha
        + 0.15 * regime_score
        + 0.12 * rs_score
        + 0.11 * risk_score
        + 0.05 * norm_sentiment
    )


def _compute_signal(
    combined_score: float, confidence: float, risk_metrics: RiskMetrics | None
) -> Signal:
    if risk_metrics and risk_metrics.is_high_risk:
        # Cap high-risk stocks at HOLD regardless of score
        if combined_score >= _AVOID_THRESHOLD:
            return Signal.HOLD
        return Signal.AVOID
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
    fundamental_score: float = 0.5,
    regime_score: float = 0.5,
    rs_score: float = 0.5,
    risk_score: float = 0.5,
    risk_metrics: RiskMetrics | None = None,
    carhart_alpha: float = 0.5,
) -> Recommendation:
    norm_sentiment = _normalize_sentiment(sentiment.avg_score)
    analyst_norm = _analyst_normalized(analyst_data)
    analyst_score = analyst_norm if analyst_norm is not None else 0.0
    pt_upside = _pt_upside_normalized(analyst_data, current_price)
    combined_score = _compute_combined(
        norm_sentiment, prediction.up_probability, analyst_norm,
        fundamental_score, regime_score, rs_score, risk_score,
        carhart_alpha, pt_upside,
    )
    signal = _compute_signal(combined_score, prediction.confidence, risk_metrics)

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
        key_news_urls=sentiment.top_news_urls,
        ticker_insight="",
        current_price=round(current_price, 2),
        price_change_pct=round(price_change_pct, 2),
        fundamental_score=round(fundamental_score, 4),
        regime_score=round(regime_score, 4),
        rs_score=round(rs_score, 4),
        risk_score=round(risk_score, 4),
        risk_metrics=risk_metrics,
        carhart_alpha=round(carhart_alpha, 4),
        pt_upside=round(pt_upside, 4),
    )


def rank_recommendations(recommendations: list[Recommendation]) -> list[Recommendation]:
    signal_order = {Signal.BUY: 0, Signal.HOLD: 1, Signal.AVOID: 2}
    return sorted(
        recommendations,
        key=lambda r: (signal_order[r.signal], -r.combined_score),
    )
