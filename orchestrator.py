from datetime import date

from loguru import logger

from config.settings import Config
from ingestion.news_fetcher import NewsFetcher
from ingestion.price_fetcher import PriceFetcher
from ingestion.analyst_fetcher import AnalystFetcher
from processing.deduplicator import process as process_news
from analysis.sentiment import SentimentAnalyser
from analysis.predictor import PricePredictor
from engine.recommender import recommend, rank_recommendations
from reporting.generator import ReportGenerator
from reporting.formatter import ReportFormatter
from delivery.email_sender import EmailSender
from models.types import ReportData


class Orchestrator:
    def __init__(self, config: Config):
        self._config = config
        self._news_fetcher = NewsFetcher()
        self._price_fetcher = PriceFetcher()
        self._analyst_fetcher = AnalystFetcher()
        self._sentiment = SentimentAnalyser()
        self._predictor = PricePredictor(config.training_window_days)
        self._report_gen = ReportGenerator(
            flash_model=config.flash_model,
            pro_model=config.pro_model,
            api_key=config.deepseek_api_key,
        )
        self._formatter = ReportFormatter(config.outputs_dir)
        self._mailer = EmailSender(
            smtp_host=config.smtp_host,
            smtp_port=config.smtp_port,
            smtp_user=config.smtp_user,
            smtp_password=config.smtp_password,
            recipient_email=config.recipient_email,
        )

    def run_pipeline(self) -> None:
        today = date.today().isoformat()
        tickers = self._config.tickers
        logger.info(f"Pipeline start — {today} — tickers: {tickers}")

        # Step 1: Ingest news
        logger.info("Step 1/9: Fetching news")
        raw_news = self._news_fetcher.fetch(tickers, self._config.news_lookback_days)

        # Step 2: Fetch prices
        logger.info("Step 2/9: Fetching price data")
        price_data = self._price_fetcher.fetch(tickers, self._config.price_lookback_days)

        # Step 3: Fetch analyst data
        logger.info("Step 3/9: Fetching analyst data")
        analyst_data = self._analyst_fetcher.fetch(tickers)

        # Step 4: Process news
        logger.info("Step 4/9: Processing news")
        news_items = process_news(raw_news, tickers)
        logger.info(f"Processed {len(news_items)} unique news items")

        # Step 5: Sentiment analysis
        logger.info("Step 5/9: Running FinBERT sentiment analysis")
        scored_items = self._sentiment.score_batch(news_items)
        ticker_sentiments = self._sentiment.aggregate_by_ticker(scored_items, tickers, today)

        # Step 6: Price prediction
        logger.info("Step 6/9: Running price predictions")
        ticker_predictions: dict = {}
        for ticker in tickers:
            df = price_data.get(ticker)
            if df is None:
                logger.warning(f"{ticker}: no price data — skipping prediction")
                continue
            sentiment_score = ticker_sentiments.get(ticker)
            avg_score = sentiment_score.avg_score if sentiment_score else 0.0
            pred = self._predictor.predict(ticker, df, avg_score)
            if pred:
                ticker_predictions[ticker] = pred

        # Step 7: Generate recommendations
        logger.info("Step 7/9: Generating recommendations")
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
            )
            recommendations.append(rec)

        ranked = rank_recommendations(recommendations)
        top_picks = [r for r in ranked if r.signal.value == "BUY"]

        # Step 8: Generate report narrative
        logger.info("Step 8/9: Generating report content with Claude")
        report_data = ReportData(
            date=today,
            market_summary="",
            recommendations=ranked,
            ticker_sentiments=ticker_sentiments,
            ticker_predictions=ticker_predictions,
            top_picks=top_picks,
            notes=(
                "Model trained on a 90-day rolling window of OHLCV data. "
                "Sentiment scored by FinBERT (ProsusAI/finbert, 0.94 F1 on Financial PhraseBank). "
                "Analyst data sourced from TipRanks where available, yfinance otherwise. "
                "Predictions are probabilistic — not financial advice."
            ),
        )

        insights = self._report_gen.generate_ticker_insights(ranked)
        for rec in report_data.recommendations:
            rec.ticker_insight = insights.get(rec.ticker, "")

        report_data.market_summary = self._report_gen.generate_market_narrative(report_data)

        # Step 9: Format, save, and email
        logger.info("Step 9/9: Formatting report and sending email")
        md_path, txt_path = self._formatter.render(report_data)

        try:
            self._mailer.send(md_path, txt_path, today)
        except Exception as exc:
            logger.error(f"Email delivery failed — {exc}. Report saved at {md_path}")

        logger.info(f"Pipeline complete — {today}")
