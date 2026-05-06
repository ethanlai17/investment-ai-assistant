# Investment AI Assistant

A production-ready AI-powered daily investment report system for a predefined list of stock tickers.

## What it does

1. **Ingests** daily financial news from Yahoo Finance RSS (yfinance fallback)
2. **Fetches** analyst consensus and price targets from TipRanks (yfinance fallback)
3. **Scores sentiment** using FinBERT — a finance-specific BERT model (0.94 F1 on Financial PhraseBank, free, local)
4. **Predicts** next-day price direction using a RandomForest trained on a 90-day rolling window of OHLCV + technical indicators
5. **Ranks** tickers with a combined score weighted across sentiment, ML prediction, and analyst consensus
6. **Generates** a report with per-ticker insights (DeepSeek Flash) and an overall market narrative (DeepSeek Pro)
7. **Emails** the report daily at 9:30 AM UK time to your configured address

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/ethanlai17/investment-ai-assistant.git
cd investment-ai-assistant
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> FinBERT (~300 MB) downloads automatically on first run.

### 3. Configure environment

Create a `.env` file in the project root and fill in:

| Variable | Description |
|----------|-------------|
| `DEEPSEEK_API_KEY` | From [platform.deepseek.com](https://platform.deepseek.com) |
| `SMTP_USER` | Your Gmail address |
| `SMTP_PASSWORD` | Gmail App Password (16 chars) — see note below |
| `RECIPIENT_EMAIL` | Where to send the daily report (defaults to `SMTP_USER`) |
| `TICKERS` | Comma-separated ticker symbols, e.g. `META,KGC,ORCL,MU` |

**Gmail App Password:** Enable 2FA on your Google account → Security → App passwords → create one for "Mail".

### 4. Run immediately (test)

```bash
python main.py --run-now
```

Output is saved to `outputs/YYYY-MM-DD.md` and `.txt`. An email is also sent if SMTP is configured.

### 5. Start the scheduler

```bash
python main.py
```

Runs daily at `SCHEDULE_HOUR:SCHEDULE_MINUTE` in `SCHEDULE_TIMEZONE` (default: 09:30 Europe/London).

## Configuration reference

All configuration is via `.env`.

| Variable | Default | Description |
|----------|---------|-------------|
| `TICKERS` | `META,KGC,ORCL,MU` | Tickers to analyse |
| `PRICE_LOOKBACK_DAYS` | `200` | Historical OHLCV window (calendar days) |
| `TRAINING_WINDOW_DAYS` | `90` | Rolling window for RandomForest training |
| `NEWS_LOOKBACK_DAYS` | `1` | Days of news to fetch |
| `SCHEDULE_HOUR` | `9` | Hour to run (24h) |
| `SCHEDULE_MINUTE` | `30` | Minute to run |
| `SCHEDULE_TIMEZONE` | `Europe/London` | Any pytz timezone string |
| `FLASH_MODEL` | `deepseek-v4-flash` | Model for per-ticker insights |
| `PRO_MODEL` | `deepseek-v4-pro` | Model for market narrative |
| `LOG_LEVEL` | `INFO` | DEBUG / INFO / WARNING |

## Architecture

```
Yahoo Finance RSS ──► NewsFetcher
yfinance news      ──► (fallback)         ┐
yfinance OHLCV     ──► PriceFetcher       │
TipRanks / yfinance──► AnalystFetcher     │
                                           │
                   Processing (clean, dedup, map)
                                           │
                   FinBERT sentiment       │ (local, free)
                   RandomForest predict    │ (60-day rolling)
                   Recommender            │ (weighted score)
                                           │
                   DeepSeek Flash ──► per-ticker insights
                   DeepSeek Pro   ──► market narrative
                                           │
                   Jinja2 templates  ──► .md + .txt report
                   Gmail SMTP_SSL    ──► email delivery
```

## Recommendation scoring

| Signal | Without analyst data | With analyst data |
|--------|---------------------|-------------------|
| Weights | 40% sentiment + 60% ML | 30% sentiment + 50% ML + 20% analyst |
| BUY | score ≥ 0.65 AND confidence ≥ 0.60 | same |
| HOLD | 0.45 ≤ score < 0.65 | same |
| AVOID | score < 0.45 | same |

## Output format

```
# Investment Report — YYYY-MM-DD

## Market Summary
[2–4 paragraph narrative from DeepSeek Pro]

## Ticker Analysis
### META — BUY
| Metric | Value |
...
Key Insight: [DeepSeek Flash one-liner]
Key News: ...

## Top Picks Today
1. META — Score: 0.712

## Notes
[model confidence and limitations]
```

## Logs

Rotating daily logs in `logs/investment_assistant_YYYY-MM-DD.log` (30-day retention).

## Limitations

- RandomForest is trained fresh daily on a 90-day rolling window — performance varies on low-liquidity tickers
- TipRanks scraping may be blocked by Cloudflare; yfinance analyst data is used as fallback
- FinBERT may not cover non-English news sources
- Not financial advice
