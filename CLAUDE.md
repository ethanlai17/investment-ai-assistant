# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the system

```bash
# One-off run (test or manual trigger)
python main.py --run-now

# Start daily scheduler (09:30 Europe/London by default)
python main.py

# Run with debug logging
LOG_LEVEL=DEBUG python main.py --run-now
```

There is no test suite yet. To validate a module in isolation, import and call it directly:

```python
# Example: test price fetcher
from config.settings import Config
from ingestion.price_fetcher import PriceFetcher
config = Config.load()
data = PriceFetcher().fetch(["META"], 30)
print(data["META"].tail())
```

## Architecture

The pipeline is driven by `orchestrator.py` → `Orchestrator.run_pipeline()`, which executes 9 sequential steps. All configuration is injected via `config/settings.py` → `Config` (loaded from `.env`). All shared dataclasses live in `models/types.py` — read this first when touching any module.

**Data flow:**

```
ingestion/ → processing/ → analysis/ → engine/ → reporting/ → delivery/
```

1. `ingestion/news_fetcher.py` — Yahoo Finance RSS (feedparser) per ticker, falls back to `yf.Ticker().news`
2. `ingestion/price_fetcher.py` — `yf.download()` for 90-day OHLCV
3. `ingestion/analyst_fetcher.py` — Scrapes TipRanks `__NEXT_DATA__` JSON; falls back to `yf.Ticker().recommendations_summary` + `analyst_price_targets`
4. `processing/deduplicator.py` — `process()` cleans, deduplicates (SequenceMatcher, threshold 0.85), and maps tickers to `NewsItem` objects
5. `analysis/sentiment.py` — FinBERT (`ProsusAI/finbert`) loaded once at startup; `score_batch()` runs all items in a single pipeline call; `aggregate_by_ticker()` returns `TickerSentiment` with `avg_score` in `[-1, +1]`
6. `analysis/predictor.py` — `PricePredictor.predict()` trains a `RandomForestClassifier` fresh on every call using a rolling 60-day window; requires ≥ 30 rows or returns `None`
7. `engine/recommender.py` — `recommend()` computes `combined_score`: `0.30×sentiment + 0.50×ML + 0.20×analyst` (when analyst data available), else `0.40×sentiment + 0.60×ML`. BUY ≥ 0.65 & confidence ≥ 0.60; HOLD ≥ 0.45; AVOID otherwise
8. `reporting/generator.py` — Two Claude calls: **Haiku** (`generate_ticker_insights`) for one-sentence per-ticker rationale (batched, JSON response); **Sonnet** (`generate_market_narrative`) for the 2–4 paragraph summary
9. `reporting/formatter.py` — Jinja2 renders `reporting/templates/report.md.j2` and `report.txt.j2` to `outputs/YYYY-MM-DD.{md,txt}`
10. `delivery/email_sender.py` — SMTP_SSL port 465; plain text + HTML (converted via `markdown` library)

## Key design constraints

- **FinBERT downloads ~300 MB on first run** and is cached by HuggingFace in `~/.cache/huggingface/`. It is loaded once in `SentimentAnalyser.__init__()` — do not instantiate it per-call.
- **RandomForest trains fresh each run** — there is no persisted model file. To swap in a different predictor, implement `predict(ticker, price_df, sentiment_score) -> Prediction | None` and replace `PricePredictor` in `orchestrator.py`.
- **All per-ticker failures are non-fatal.** The orchestrator logs a warning and skips tickers missing price data, predictions, or sentiment. The report still generates for successful tickers.
- `Recommendation.ticker_insight` is intentionally left empty (`""`) by `recommender.py` and populated later by `ReportGenerator.generate_ticker_insights()` in the orchestrator before `generate_market_narrative()` is called.
- The `_FEATURE_COLS` list in `analysis/features.py` defines the canonical feature order. `build_training_set()` and `build_prediction_row()` must produce columns in the same order or the scaler will misalign.

## Environment

Copy `.env.example` to `.env`. Required: `ANTHROPIC_API_KEY`, `SMTP_USER`, `SMTP_PASSWORD`. Everything else has defaults. `SMTP_PASSWORD` must be a Gmail App Password (16 chars), not an account password.
