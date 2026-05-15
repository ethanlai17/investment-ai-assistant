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
data = PriceFetcher().fetch(["AAPL"], 30)
print(data["AAPL"].tail())
```

## Architecture

The pipeline is driven by `orchestrator.py` → `Orchestrator.run_pipeline()`, which executes 17 sequential steps. All configuration is injected via `config/settings.py` → `Config` (loaded from `.env`). All shared dataclasses live in `models/types.py` — read this first when touching any module.

**Data flow:**

```
ingestion/ → processing/ → analysis/ → engine/ → reporting/ → delivery/
```

1. `ingestion/news_fetcher.py` — Yahoo Finance RSS (feedparser) per ticker, falls back to `yf.Ticker().news`
2. `ingestion/price_fetcher.py` — `yf.download()` for 500-day OHLCV
3. `ingestion/analyst_fetcher.py` — Scrapes TipRanks `__NEXT_DATA__` JSON; falls back to `yf.Ticker().recommendations_summary` + `analyst_price_targets`
4. `ingestion/fundamental_fetcher.py` — yfinance `Ticker.info`: PE, PB, ROE, ROA, FCF yield, earnings growth, D/E, current ratio, operating margin
5. `ingestion/macro_fetcher.py` — yfinance batch download: SPY returns, VIX levels, 10Y yield (^TNX), 13W T-bill (^IRX)
6. `ingestion/sector_etf_fetcher.py` — Maps tickers to GICS sectors and corresponding ETFs (XLK, XLF, XLV…)
7. `processing/deduplicator.py` — `process()` cleans, deduplicates (SequenceMatcher, threshold 0.85), and maps tickers to `NewsItem` objects
8. `analysis/sentiment.py` — FinBERT (`ProsusAI/finbert`) loaded once at startup; `score_batch()` runs all items in a single pipeline call; `aggregate_by_ticker()` returns `TickerSentiment` with `avg_score` in `[-1, +1]`
9. `analysis/regime.py` — 2-state Gaussian HMM on SPY returns; `regime_score = 0.60×HMM_bull_prob + 0.25×VIX_signal + 0.15×yield_curve_signal`
10. `analysis/fundamental.py` — 8-factor Fama-French-style cross-sectional percentile rank: earnings yield, book yield, ROE, op margin, FCF yield, earnings growth, D/E (inverse), current ratio
11. `analysis/relative_strength.py` — RS ratio (stock / sector ETF cumulative return) over 13/26/52 weeks; `rs_score = 0.40×13W + 0.35×26W + 0.25×52W rank`
12. `analysis/risk_metrics.py` — Per-ticker: Sharpe, Sortino (downside deviation only), CVaR 95% (tail loss), MDD, beta. `risk_score = 0.60×sortino_rank + 0.40×sharpe_rank − cvar_penalty`. High-risk flag: CVaR < −5% daily OR beta > 2.0 (caps signal at HOLD; overrides risk label to "high risk (CVaR/beta breach)" regardless of score)
13. `analysis/factor_decomposition.py` — Carhart 4-factor OLS regression (MKT-RF, SMB, HML, UMD) using Fama-French daily factors fetched from Kenneth French's data library. `carhart_alpha` = cross-sectional rank of annualised OLS intercept
14. `analysis/predictor.py` — `PricePredictor.predict()` trains a `HistGradientBoostingClassifier` fresh on every call using a rolling 252-day window; 15 features (OHLCV-derived technicals + composite scores + 52-week high proximity); requires ≥ 30 rows or returns `None`
15. `engine/recommender.py` — `recommend()` computes 9-factor `combined_score` (see weights below). BUY ≥ 0.65 & confidence ≥ 0.60; HOLD ≥ 0.45; AVOID otherwise. High-risk stocks capped at HOLD
16. `reporting/generator.py` — Two DeepSeek calls via OpenAI-compatible API: **Flash model** (`generate_ticker_insights`) for one-sentence per-ticker rationale (batched, JSON response); **Pro model** (`generate_market_narrative`) for the 2–4 paragraph summary
17. `reporting/formatter.py` — Jinja2 renders `reporting/templates/report.md.j2` and `report.txt.j2` to `outputs/YYYY-MM-DD.{md,txt}`
18. `delivery/email_sender.py` — SMTP_SSL port 465; plain text + HTML (converted via `markdown` library)

## Composite Score Weights

```
With analyst data:
  0.16 × fundamental_score
  0.18 × ml_up_probability
  0.12 × carhart_alpha       ← Carhart 4-factor OLS alpha, cross-sectional rank
  0.12 × regime_score
  0.10 × rs_score
  0.09 × risk_score          ← Sortino/CVaR-weighted
  0.10 × sentiment_score
  0.07 × analyst_norm
  0.06 × pt_upside           ← analyst price target implied upside, normalised

Without analyst data:
  0.20 × fundamental_score
  0.22 × ml_up_probability
  0.15 × carhart_alpha
  0.15 × regime_score
  0.12 × rs_score
  0.11 × risk_score
  0.05 × sentiment_score
```

## Key design constraints

- **FinBERT downloads ~300 MB on first run** and is cached by HuggingFace in `~/.cache/huggingface/`. It is loaded once in `SentimentAnalyser.__init__()` — do not instantiate it per-call.
- **HistGradientBoostingClassifier trains fresh each run** — there is no persisted model file. To swap in a different predictor, implement `predict(ticker, price_df, ...) -> Prediction | None` and replace `PricePredictor` in `orchestrator.py`.
- **Carhart factors are fetched once per run** from Kenneth French's data library (HTTP, no API key). If the fetch fails, all tickers get `carhart_alpha=0.5` (neutral) and the pipeline continues.
- **All per-ticker failures are non-fatal.** The orchestrator logs a warning and skips tickers missing price data, predictions, or sentiment. The report still generates for successful tickers.
- `Recommendation.ticker_insight` is intentionally left empty (`""`) by `recommender.py` and populated later by `ReportGenerator.generate_ticker_insights()` in the orchestrator before `generate_market_narrative()` is called.
- The `_FEATURE_COLS` list in `analysis/features.py` defines the canonical 15-feature order. `build_training_set()` and `build_prediction_row()` must produce columns in the same order or the model will misalign.
- **S&P 500 scan** calls `run_pipeline(top_n_buys=5)` which returns the top 5 tickers prioritised by signal: BUY tickers ranked by `combined_score` first, then HOLD tickers by `combined_score` to fill any remaining slots. AVOID tickers are never included. The daily watchlist report returns all tickers.

## Environment

Create `.env` in the project root. Required: `DEEPSEEK_API_KEY`, `SMTP_USER`, `SMTP_PASSWORD`, `TICKERS`. Everything else has defaults. `SMTP_PASSWORD` must be a Gmail App Password (16 chars), not an account password.
