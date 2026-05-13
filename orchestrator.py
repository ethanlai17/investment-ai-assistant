from datetime import date

from loguru import logger

from config.settings import Config
from ingestion.news_fetcher import NewsFetcher
from ingestion.price_fetcher import PriceFetcher
from ingestion.analyst_fetcher import AnalystFetcher
from ingestion.fundamental_fetcher import FundamentalFetcher
from ingestion.macro_fetcher import MacroFetcher
from ingestion.sector_etf_fetcher import SectorETFFetcher
from ingestion.reference_builder import ReferenceDistributionBuilder
from processing.deduplicator import process as process_news
from analysis.sentiment import SentimentAnalyser
from analysis.fundamental import FundamentalScorer
from analysis.regime import RegimeDetector
from analysis.relative_strength import RelativeStrengthCalculator
from analysis.risk_metrics import RiskCalculator
from analysis.predictor import PricePredictor
from engine.recommender import recommend, rank_recommendations
from reporting.generator import ReportGenerator
from reporting.formatter import ReportFormatter
from delivery.email_sender import EmailSender
from models.types import ReportData, RegimeState


class Orchestrator:
    def __init__(self, config: Config):
        self._config = config
        self._news_fetcher = NewsFetcher()
        self._price_fetcher = PriceFetcher()
        self._analyst_fetcher = AnalystFetcher()
        self._fundamental_fetcher = FundamentalFetcher()
        self._macro_fetcher = MacroFetcher()
        self._sector_fetcher = SectorETFFetcher()
        self._sentiment = SentimentAnalyser()
        self._fundamental_scorer = FundamentalScorer()
        self._regime_detector = RegimeDetector()
        self._rs_calculator = RelativeStrengthCalculator()
        self._risk_calculator = RiskCalculator()
        self._predictor = PricePredictor(config.training_window_days)
        self._report_gen = ReportGenerator(
            flash_model=config.flash_model,
            pro_model=config.pro_model,
            api_key=config.deepseek_api_key,
        )
        self._ref_builder = ReferenceDistributionBuilder()
        self._formatter = ReportFormatter(config.outputs_dir)
        self._mailer = EmailSender(
            smtp_host=config.smtp_host,
            smtp_port=config.smtp_port,
            smtp_user=config.smtp_user,
            smtp_password=config.smtp_password,
            recipient_email=config.recipient_email,
        )

    def run_sp500_scan(self) -> None:
        from ingestion.sp500_screener import SP500Screener
        logger.info("Starting S&P 500 scan")
        candidates = SP500Screener().screen(top_n=50)
        self.run_pipeline(
            tickers=candidates,
            top_n_buys=5,
            report_suffix="-sp500-scan",
            report_title="S&P 500 Top 5 Picks",
        )

    def run_pipeline(
        self,
        tickers: list[str] | None = None,
        top_n_buys: int | None = None,
        report_suffix: str = "",
        report_title: str = "Investment Report",
    ) -> None:
        today = date.today().isoformat()
        tickers = tickers or self._config.tickers
        logger.info(f"Pipeline start — {today} — tickers: {tickers}")

        # Step 0: Load S&P 500 reference distributions (cached per day)
        logger.info("Step 0: Loading S&P 500 reference distributions")
        ref = None
        try:
            ref = self._ref_builder.build(self._config.outputs_dir)
        except Exception as exc:
            logger.warning(f"Reference distribution build failed — {exc}. Falling back to in-batch ranking.")

        # Step 1: Fetch news
        logger.info("Step 1/16: Fetching news")
        raw_news = self._news_fetcher.fetch(tickers, self._config.news_lookback_days)

        # Step 2: Fetch prices
        logger.info("Step 2/16: Fetching price data")
        price_data = self._price_fetcher.fetch(tickers, self._config.price_lookback_days)

        # Step 3: Fetch analyst data
        logger.info("Step 3/16: Fetching analyst data")
        analyst_data = self._analyst_fetcher.fetch(tickers)

        # Step 4: Fetch fundamental data
        logger.info("Step 4/16: Fetching fundamental data")
        fundamental_data = self._fundamental_fetcher.fetch(tickers)

        # Step 5: Fetch macro data
        logger.info("Step 5/16: Fetching macro data (SPY, VIX, yield curve)")
        macro_data = self._macro_fetcher.fetch()

        # Step 6: Fetch sector ETF data
        logger.info("Step 6/16: Fetching sector ETF data")
        ticker_sector, ticker_etf, etf_prices = self._sector_fetcher.fetch(tickers)

        # Step 7: Process news
        logger.info("Step 7/16: Processing news")
        news_items = process_news(raw_news, tickers)
        logger.info(f"Processed {len(news_items)} unique news items")

        # Step 8: Sentiment analysis
        logger.info("Step 8/16: Running FinBERT sentiment analysis")
        scored_items = self._sentiment.score_batch(news_items)
        ticker_sentiments = self._sentiment.aggregate_by_ticker(scored_items, tickers, today)

        # Step 9: Macro regime detection
        logger.info("Step 9/16: Detecting market regime")
        regime: RegimeState | None = None
        if macro_data is not None:
            regime = self._regime_detector.detect(macro_data)
            logger.info(
                f"Regime: {regime.current_state}, score={regime.regime_score:.4f}, "
                f"VIX={macro_data.current_vix:.1f}, spread={macro_data.current_spread:.2f}"
            )
        else:
            logger.warning("Macro data unavailable — skipping regime detection")

        # Step 10: Fundamental scoring
        logger.info("Step 10/16: Scoring fundamental factors")
        fundamental_scores = self._fundamental_scorer.score(fundamental_data)

        # Step 11: Relative strength
        logger.info("Step 11/16: Computing sector-relative strength")
        price_data_available = {t: df for t, df in price_data.items() if df is not None}
        rs_data = self._rs_calculator.compute_all(
            price_data_available, ticker_etf, ticker_sector, etf_prices, reference=ref
        )

        # Step 12: Risk metrics
        logger.info("Step 12/16: Computing risk metrics")
        spy_returns = macro_data.spy_returns if macro_data is not None else None
        risk_free_rate = 0.04  # ~4% annualised T-bill, fixed conservative fallback
        risk_metrics_all = {}
        if spy_returns is not None:
            risk_metrics_all = self._risk_calculator.compute_all(
                price_data_available, spy_returns, risk_free_rate,
                reference_sharpes=ref["sharpes"] if ref is not None else None,
            )

        # Step 13: Price prediction
        logger.info("Step 13/16: Running price predictions")
        ticker_predictions: dict = {}
        regime_score_val = regime.regime_score if regime is not None else 0.5
        for ticker in tickers:
            df = price_data.get(ticker)
            if df is None:
                logger.warning(f"{ticker}: no price data — skipping prediction")
                continue
            sentiment_score = ticker_sentiments.get(ticker)
            avg_score = sentiment_score.avg_score if sentiment_score else 0.0
            pred = self._predictor.predict(
                ticker=ticker,
                price_df=df,
                sentiment_score=avg_score,
                fundamental_score=fundamental_scores.get(ticker, 0.5),
                regime_score=regime_score_val,
                rs_score=rs_data[ticker].rs_score if ticker in rs_data else 0.5,
                risk_score=risk_metrics_all[ticker].risk_score if ticker in risk_metrics_all else 0.5,
            )
            if pred:
                ticker_predictions[ticker] = pred

        # Step 14: Generate recommendations
        logger.info("Step 14/16: Generating recommendations")
        recommendations = []
        for ticker in tickers:
            sentiment = ticker_sentiments.get(ticker)
            prediction = ticker_predictions.get(ticker)
            if sentiment is None or prediction is None:
                logger.warning(f"{ticker}: missing sentiment or prediction — skipping")
                continue

            df = price_data.get(ticker)
            current_price = 0.0
            price_change_pct = 0.0
            if df is not None and not df.empty:
                close = df["Close"].squeeze()
                current_price = float(close.iloc[-1])
                if len(close) >= 2:
                    price_change_pct = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)

            rec = recommend(
                ticker=ticker,
                sentiment=sentiment,
                prediction=prediction,
                analyst_data=analyst_data.get(ticker),
                current_price=current_price,
                price_change_pct=price_change_pct,
                fundamental_score=fundamental_scores.get(ticker, 0.5),
                regime_score=regime_score_val,
                rs_score=rs_data[ticker].rs_score if ticker in rs_data else 0.5,
                risk_score=risk_metrics_all[ticker].risk_score if ticker in risk_metrics_all else 0.5,
                risk_metrics=risk_metrics_all.get(ticker),
            )
            recommendations.append(rec)

        ranked = rank_recommendations(recommendations)
        top_picks = [r for r in ranked if r.signal.value == "BUY"]

        if top_n_buys is not None:
            top_picks = sorted(top_picks, key=lambda r: r.combined_score, reverse=True)[:top_n_buys]
            ranked = list(top_picks)

        # Step 15: Generate report narrative
        logger.info("Step 15/16: Generating report content with DeepSeek")
        regime_note = ""
        if regime:
            regime_note = (
                f"Market regime: {regime.current_state} "
                f"(bull_prob={regime.bull_probability:.2f}, "
                f"VIX={macro_data.current_vix:.1f}, "  # type: ignore[union-attr]
                f"yield_spread={macro_data.current_spread:.2f}pp). "  # type: ignore[union-attr]
            )

        scan_note = (
            f"S&P 500 scan: pre-filtered from 503 constituents via momentum + fundamentals. "
            f"Top {top_n_buys} BUY signals shown, ranked by combined score. "
            if top_n_buys is not None else ""
        )
        report_data = ReportData(
            date=today,
            market_summary="",
            recommendations=ranked,
            ticker_sentiments=ticker_sentiments,
            ticker_predictions=ticker_predictions,
            top_picks=top_picks,
            notes=(
                f"{scan_note}{regime_note}"
                "XGBoost trained on 252-day rolling window; target = 20-day forward return. "
                "Fundamental scoring: Fama-French-style 8-factor cross-sectional rank (value, quality, growth, safety). "
                "Regime: HMM (2-state) on SPY returns + VIX + yield curve. "
                "Relative strength: 13/26/52-week rank vs sector ETF. "
                "Risk: Sharpe/drawdown/beta cross-sectional rank; high-risk stocks capped at HOLD. "
                "Sentiment: FinBERT (ProsusAI/finbert, 0.94 F1). "
                "Analyst data: TipRanks (yfinance fallback). "
                "Predictions are probabilistic — not financial advice."
            ),
        )

        insights = self._report_gen.generate_ticker_insights(ranked, regime, ticker_sector)
        for rec in report_data.recommendations:
            rec.ticker_insight = insights.get(rec.ticker, "")

        report_data.market_summary = self._report_gen.generate_market_narrative(report_data, regime)

        # Step 16: Format, save, and email
        logger.info("Step 16/16: Formatting report and sending email")
        md_path, txt_path = self._formatter.render(report_data, suffix=report_suffix, report_title=report_title)

        email_subject = (
            f"{report_title} — {today}" if report_suffix else None
        )
        try:
            self._mailer.send(md_path, txt_path, today, subject=email_subject)
        except Exception as exc:
            logger.error(f"Email delivery failed — {exc}. Report saved at {md_path}")

        logger.info(f"Pipeline complete — {today}")
