import ast
import json
import re

from openai import OpenAI
from loguru import logger

from models.types import Recommendation, ReportData, RegimeState

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class ReportGenerator:
    def __init__(self, flash_model: str, pro_model: str, api_key: str):
        self._client = OpenAI(api_key=api_key, base_url=_DEEPSEEK_BASE_URL)
        self._flash = flash_model
        self._pro = pro_model

    def generate_ticker_insights(
        self,
        recommendations: list[Recommendation],
        regime: RegimeState | None = None,
        ticker_sector: dict[str, str] | None = None,
    ) -> dict[str, str]:
        if not recommendations:
            return {}

        regime_ctx = ""
        if regime:
            regime_ctx = (
                f"Current market regime: {regime.current_state} "
                f"(bull_probability={regime.bull_probability:.2f}, "
                f"VIX_signal={regime.vix_signal:.2f}, "
                f"yield_curve_signal={regime.yield_curve_signal:.2f}). "
            )

        ticker_data = []
        for rec in recommendations:
            analyst_str = "N/A"
            if rec.analyst_data and rec.analyst_data.consensus != "Unknown":
                analyst_str = (
                    f"{rec.analyst_data.consensus} "
                    f"(target ${rec.analyst_data.price_target:.2f}, "
                    f"{rec.analyst_data.analyst_count} analysts)"
                )
            risk_str = "N/A"
            if rec.risk_metrics:
                rm = rec.risk_metrics
                risk_str = (
                    f"Sharpe={rm.sharpe_ratio:.2f}, Sortino={rm.sortino_ratio:.2f}, "
                    f"CVaR95={rm.cvar_95:.2%}, MaxDD={rm.max_drawdown:.1%}, "
                    f"Beta={rm.beta:.2f}, HighRisk={rm.is_high_risk}"
                )
            sector = (ticker_sector or {}).get(rec.ticker, "Unknown")
            ticker_data.append({
                "ticker": rec.ticker,
                "sector": sector,
                "signal": rec.signal.value,
                "combined_score": rec.combined_score,
                "fundamental_score": rec.fundamental_score,
                "regime_score": rec.regime_score,
                "rs_score": rec.rs_score,
                "risk_score": rec.risk_score,
                "carhart_alpha": rec.carhart_alpha,
                "pt_upside": rec.pt_upside,
                "sentiment_score": rec.sentiment_score,
                "ml_up_probability": rec.ml_up_probability,
                "analyst_consensus": analyst_str,
                "risk_metrics": risk_str,
                "top_headlines": rec.key_news,
            })

        prompt = (
            f"{regime_ctx}"
            "You are explaining stocks to someone with no financial background. "
            "For each ticker below, write exactly ONE plain-English sentence that: "
            "(1) briefly describes what the stock has been doing recently (e.g. rising, falling, or flat), and "
            "(2) states clearly whether to hold, buy, or avoid it and why in everyday language. "
            "No jargon, no technical terms, no metric names. Write as if texting a friend. "
            "Return ONLY a JSON object with ticker symbols as keys and one-sentence insights as values.\n\n"
            f"Ticker data:\n{json.dumps(ticker_data, indent=2)}"
        )

        try:
            response = self._client.chat.completions.create(
                model=self._flash,
                max_tokens=4096,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content or ""
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                text = match.group(0)
            try:
                insights: dict[str, str] = json.loads(text)
            except json.JSONDecodeError:
                insights = ast.literal_eval(text)
            logger.debug(f"Flash insights generated for {list(insights.keys())}")
            return insights
        except Exception as exc:
            logger.warning(f"Flash ticker insights failed — {exc}")
            return {rec.ticker: "" for rec in recommendations}

    def generate_market_narrative(
        self,
        report_data: ReportData,
        regime: RegimeState | None = None,
    ) -> str:
        market_mood = "positive" if (regime and regime.bull_probability >= 0.6) else "cautious"
        market_calm = "calm" if (regime and regime.vix_signal >= 0.6) else "volatile"

        recs_summary = []
        for rec in report_data.recommendations:
            price_dir = "up" if rec.price_change_pct > 0 else "down"
            analyst_str = ""
            if rec.analyst_data and rec.analyst_data.consensus != "Unknown":
                analyst_str = f", experts say {rec.analyst_data.consensus}"
            risky = " (risky)" if rec.risk_metrics and rec.risk_metrics.is_high_risk else ""
            headline = rec.key_news[0] if rec.key_news else ""
            recs_summary.append(
                f"{rec.ticker}{risky}: signal={rec.signal.value}, "
                f"price {price_dir} {abs(rec.price_change_pct):.1f}% today, "
                f"news mood={'positive' if rec.sentiment_score > 0.6 else 'mixed' if rec.sentiment_score >= 0.4 else 'negative'}, "
                f"AI predicts {'likely up' if rec.ml_up_probability > 0.65 else 'likely down' if rec.ml_up_probability < 0.35 else 'uncertain'} in 3-4 weeks"
                f"{analyst_str}. Latest news: {headline}"
            )

        top_picks_str = ", ".join(r.ticker for r in report_data.top_picks) or "none"

        prompt = (
            f"You are explaining today's stock market ({report_data.date}) to someone with no financial background.\n\n"
            f"Market conditions today: {market_mood} overall, volatility is {market_calm}.\n"
            f"Stocks with a BUY signal today: {top_picks_str}.\n\n"
            "Ticker data:\n" + "\n".join(recs_summary) + "\n\n"
            "Write exactly 2 short paragraphs:\n"
            "1. What the overall market feels like today and whether it is a good or cautious time to invest — in plain everyday language.\n"
            "2. Highlight 2-3 notable stocks: which ones are doing well, which to be careful about, and why — no numbers, no codes, just simple observations.\n\n"
            "Rules: no jargon, no technical terms, no metric names, no score numbers. "
            "Write as if explaining to a curious friend over coffee. Keep it short and clear."
        )

        try:
            response = self._client.chat.completions.create(
                model=self._pro,
                max_tokens=2048,
                temperature=0.4,
                messages=[{"role": "user", "content": prompt}],
            )
            msg = response.choices[0].message
            narrative = (msg.content or "").strip()
            logger.debug("Pro market narrative generated")
            return narrative or "Market narrative unavailable."
        except Exception as exc:
            logger.warning(f"Pro market narrative failed — {exc}")
            return "Market narrative unavailable."
