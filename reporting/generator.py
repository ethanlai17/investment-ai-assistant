import json

import anthropic
from loguru import logger

from models.types import Recommendation, ReportData


class ReportGenerator:
    def __init__(self, haiku_model: str, sonnet_model: str, api_key: str):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._haiku = haiku_model
        self._sonnet = sonnet_model

    def generate_ticker_insights(
        self, recommendations: list[Recommendation]
    ) -> dict[str, str]:
        """
        Single Claude Haiku call for all tickers.
        Returns {ticker: one_sentence_insight}.
        """
        if not recommendations:
            return {}

        ticker_data = []
        for rec in recommendations:
            analyst_str = "N/A"
            if rec.analyst_data and rec.analyst_data.consensus != "Unknown":
                analyst_str = (
                    f"{rec.analyst_data.consensus} "
                    f"(target ${rec.analyst_data.price_target:.2f}, "
                    f"{rec.analyst_data.analyst_count} analysts)"
                )
            ticker_data.append({
                "ticker": rec.ticker,
                "signal": rec.signal.value,
                "sentiment_score": rec.sentiment_score,
                "ml_up_probability": rec.ml_up_probability,
                "analyst_consensus": analyst_str,
                "top_headlines": rec.key_news,
            })

        prompt = (
            "You are a concise financial analyst. For each ticker below, write exactly ONE sentence "
            "summarising the key investment rationale based on the provided signals. "
            "Be specific, factual, and avoid generic phrases. "
            "Return ONLY a JSON object with ticker symbols as keys and one-sentence insights as values.\n\n"
            f"Ticker data:\n{json.dumps(ticker_data, indent=2)}"
        )

        try:
            response = self._client.messages.create(
                model=self._haiku,
                max_tokens=512,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            insights: dict[str, str] = json.loads(text)
            logger.debug(f"Haiku insights generated for {list(insights.keys())}")
            return insights
        except Exception as exc:
            logger.warning(f"Haiku ticker insights failed — {exc}")
            return {rec.ticker: "" for rec in recommendations}

    def generate_market_narrative(self, report_data: ReportData) -> str:
        """
        Claude Sonnet call for the overall market narrative.
        Returns 2–4 paragraphs of market summary.
        """
        recs_summary = []
        for rec in report_data.recommendations:
            analyst_str = "No analyst data"
            if rec.analyst_data and rec.analyst_data.consensus != "Unknown":
                analyst_str = (
                    f"Analyst consensus: {rec.analyst_data.consensus}, "
                    f"target ${rec.analyst_data.price_target:.2f}"
                )
            recs_summary.append(
                f"{rec.ticker}: {rec.signal.value}, sentiment={rec.sentiment_score:.2f}, "
                f"ML_up={rec.ml_up_probability:.2f}, combined={rec.combined_score:.2f}. "
                f"{analyst_str}. "
                f"Top headline: {rec.key_news[0] if rec.key_news else 'none'}"
            )

        top_picks_str = ", ".join(r.ticker for r in report_data.top_picks) or "None"

        prompt = (
            f"You are a senior equity analyst writing a daily morning briefing for {report_data.date}.\n\n"
            "Write 2–4 concise paragraphs summarising today's market signals across the portfolio. "
            "Connect themes across tickers where relevant. Note any divergences between ML signals and analyst consensus. "
            "Be direct and specific — no filler phrases.\n\n"
            f"Top picks today: {top_picks_str}\n\n"
            "Ticker signals:\n" + "\n".join(recs_summary)
        )

        try:
            response = self._client.messages.create(
                model=self._sonnet,
                max_tokens=600,
                temperature=0.4,
                messages=[{"role": "user", "content": prompt}],
            )
            narrative = response.content[0].text.strip()
            logger.debug("Sonnet market narrative generated")
            return narrative
        except Exception as exc:
            logger.warning(f"Sonnet market narrative failed — {exc}")
            return "Market narrative unavailable."
