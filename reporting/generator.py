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
                    f"Sharpe={rm.sharpe_ratio:.2f}, MaxDD={rm.max_drawdown:.1%}, "
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
                "sentiment_score": rec.sentiment_score,
                "ml_up_probability": rec.ml_up_probability,
                "analyst_consensus": analyst_str,
                "risk_metrics": risk_str,
                "top_headlines": rec.key_news,
            })

        prompt = (
            f"{regime_ctx}"
            "You are a concise financial analyst. For each ticker below, write exactly ONE sentence "
            "summarising the key investment rationale based on all provided signals. "
            "Lead with the dominant factor (fundamental quality, macro regime, sector strength, or risk). "
            "Be specific and factual. For HIGH_RISK tickers, mention the risk explicitly. "
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
        regime_block = ""
        if regime:
            regime_block = (
                f"MACRO REGIME: {regime.current_state.upper()} "
                f"(bull_prob={regime.bull_probability:.2f}, "
                f"VIX_signal={regime.vix_signal:.2f} [1=calm/0=fear], "
                f"yield_curve_signal={regime.yield_curve_signal:.2f} [1=normal/0=inverted])\n\n"
            )

        recs_summary = []
        for rec in report_data.recommendations:
            analyst_str = "No analyst data"
            if rec.analyst_data and rec.analyst_data.consensus != "Unknown":
                analyst_str = (
                    f"Analyst: {rec.analyst_data.consensus}, "
                    f"target ${rec.analyst_data.price_target:.2f}"
                )
            risk_flag = " [HIGH RISK]" if rec.risk_metrics and rec.risk_metrics.is_high_risk else ""
            recs_summary.append(
                f"{rec.ticker}{risk_flag}: {rec.signal.value}, "
                f"fundamental={rec.fundamental_score:.2f}, "
                f"regime={rec.regime_score:.2f}, "
                f"rs={rec.rs_score:.2f}, "
                f"risk={rec.risk_score:.2f}, "
                f"sentiment={rec.sentiment_score:.2f}, "
                f"ML_up={rec.ml_up_probability:.2f}, "
                f"combined={rec.combined_score:.2f}. "
                f"{analyst_str}. "
                f"Top headline: {rec.key_news[0] if rec.key_news else 'none'}"
            )

        top_picks_str = ", ".join(r.ticker for r in report_data.top_picks) or "None"

        prompt = (
            f"You are a senior equity analyst writing a daily morning briefing for {report_data.date}.\n\n"
            f"{regime_block}"
            "Write 3–5 concise paragraphs covering:\n"
            "1. The current macro regime and what it means for equity positioning.\n"
            "2. Sector rotation observations — which sectors show relative strength.\n"
            "3. Top fundamental picks — stocks with strong value/quality/growth scores.\n"
            "4. Risk warnings — flag any high-beta or high-drawdown names.\n"
            "5. Overall portfolio bias (bullish/neutral/defensive) and key risks to watch.\n\n"
            "Connect themes across tickers. Be direct and specific — no filler phrases.\n\n"
            f"Top picks today: {top_picks_str}\n\n"
            "Ticker signals:\n" + "\n".join(recs_summary)
        )

        try:
            response = self._client.chat.completions.create(
                model=self._pro,
                max_tokens=2048,
                temperature=0.4,
                messages=[{"role": "user", "content": prompt}],
            )
            msg = response.choices[0].message
            narrative = (msg.content or getattr(msg, "reasoning_content", "") or "").strip()
            logger.debug("Pro market narrative generated")
            return narrative or "Market narrative unavailable."
        except Exception as exc:
            logger.warning(f"Pro market narrative failed — {exc}")
            return "Market narrative unavailable."
