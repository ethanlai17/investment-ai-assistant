# Investment AI Assistant

A production-ready AI-powered daily investment report system for a predefined list of US stock tickers. Implements institutional-grade multi-factor analysis inspired by Fama-French factor models, Hidden Markov Model regime detection, and sector-relative strength used by quant hedge funds.

## What it does

1. **Ingests** daily financial news from Yahoo Finance RSS (yfinance fallback)
2. **Fetches** analyst consensus and price targets from TipRanks (yfinance fallback)
3. **Fetches fundamental data** — P/E, P/B, ROE, FCF yield, earnings growth, debt ratios — from yfinance
4. **Fetches macro data** — SPY, VIX, 10Y/2Y yield curve — for market regime detection
5. **Scores sentiment** using FinBERT — a finance-specific BERT model (0.94 F1 on Financial PhraseBank, free, local)
6. **Detects macro regime** using a 2-state Hidden Markov Model on SPY returns + VIX + yield curve spread
7. **Scores fundamental quality** using a Fama-French-style 8-factor cross-sectional rank (value, profitability, growth, safety)
8. **Measures sector-relative strength** vs sector ETF (XLK, XLF, XLV…) over 13/26/52-week windows
9. **Computes risk metrics** — Sharpe ratio, max drawdown, beta — cross-sectionally ranked; high-risk stocks capped at HOLD
10. **Predicts 20-day forward price direction** using HistGradientBoostingClassifier trained on a 252-day rolling window of OHLCV + technical + factor features
11. **Ranks** tickers with a 7-factor combined score weighted for long-term investing
12. **Generates** a report with per-ticker insights (DeepSeek Flash) and a macro-led market narrative (DeepSeek Pro)
13. **Emails** the report daily at 9:30 AM UK time to your configured address

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

> FinBERT (~300 MB) downloads automatically on first run and is cached in `~/.cache/huggingface/`.

### 3. Configure environment

Create a `.env` file in the project root:

| Variable | Description |
|----------|-------------|
| `DEEPSEEK_API_KEY` | From [platform.deepseek.com](https://platform.deepseek.com) |
| `SMTP_USER` | Your Gmail address |
| `SMTP_PASSWORD` | Gmail App Password (16 chars) — see note below |
| `RECIPIENT_EMAIL` | Where to send the daily report (defaults to `SMTP_USER`) |
| `TICKERS` | Comma-separated US ticker symbols, e.g. `AAPL,MSFT,NVDA` |

**Gmail App Password:** Enable 2FA on your Google account → Security → App passwords → create one for "Mail".

### 4. Run immediately (test)

```bash
./run.sh --run-now
```

Output is saved to `outputs/YYYY-MM-DD.md` and `.txt`. An email is also sent if SMTP is configured.

### 5. Start the scheduler

```bash
nohup ./run.sh >> logs/scheduler.log 2>&1 &
```

Runs daily at `SCHEDULE_HOUR:SCHEDULE_MINUTE` in `SCHEDULE_TIMEZONE` (default: 09:30 Europe/London). Survives terminal close.

To check the scheduler is alive:

```bash
pgrep -fl "python -B main.py"
```

To stop the scheduler:

```bash
pkill -f "python -B main.py"
```

## Configuration reference

| Variable | Default | Description |
|----------|---------|-------------|
| `TICKERS` | *(required)* | Comma-separated US tickers to analyse |
| `PRICE_LOOKBACK_DAYS` | `500` | Historical OHLCV window (calendar days) |
| `TRAINING_WINDOW_DAYS` | `252` | Rolling window for ML model training |
| `NEWS_LOOKBACK_DAYS` | `1` | Days of news to fetch |
| `SCHEDULE_HOUR` | `9` | Hour to run (24h) |
| `SCHEDULE_MINUTE` | `30` | Minute to run |
| `SCHEDULE_TIMEZONE` | `Europe/London` | Any pytz timezone string |
| `FLASH_MODEL` | `deepseek-v4-pro` | Model for per-ticker insights |
| `PRO_MODEL` | `deepseek-v4-pro` | Model for market narrative |
| `LOG_LEVEL` | `INFO` | DEBUG / INFO / WARNING |

## Architecture

```
Yahoo Finance RSS  ──► NewsFetcher
yfinance news      ──► (fallback)            ┐
yfinance OHLCV     ──► PriceFetcher          │
TipRanks/yfinance  ──► AnalystFetcher        │
yfinance info      ──► FundamentalFetcher    │  ingestion/
SPY/VIX/yields     ──► MacroFetcher          │
Sector ETFs        ──► SectorETFFetcher      ┘

                   Processing (clean, dedup, map tickers)

                   FinBERT           ──► sentiment scores      ┐
                   HMM (2-state)     ──► regime score          │
                   Fama-French rank  ──► fundamental score     │  analysis/
                   RS vs sector ETF  ──► rs_score              │
                   Sharpe/β/drawdown ──► risk_score            │
                   HistGBM (252-day) ──► 20-day prediction     ┘

                   7-factor Recommender ──► BUY / HOLD / AVOID

                   DeepSeek Flash ──► per-ticker insights
                   DeepSeek Pro   ──► macro-led market narrative

                   Jinja2 templates ──► .md + .txt report
                   Gmail SMTP_SSL   ──► email delivery
```

## 7-Factor Recommendation Scoring

| Factor | With analyst data | Without analyst data |
|--------|-------------------|---------------------|
| Fundamental (Fama-French) | 20% | 23% |
| ML 20-day prediction | 18% | 20% |
| Macro regime (HMM) | 15% | 17% |
| Sector-relative strength | 12% | 15% |
| Risk-adjusted score | 12% | 13% |
| Sentiment (FinBERT) | 13% | 12% |
| Analyst consensus | 10% | — |

| Signal | Condition |
|--------|-----------|
| **BUY** | combined ≥ 0.65 AND model confidence ≥ 0.60 AND not high-risk |
| **HOLD** | combined ≥ 0.45 (or high-risk stock regardless of score) |
| **AVOID** | combined < 0.45 |

High-risk = max drawdown < −30% OR beta > 2.0 in the past 252 days.

## Factor Definitions

| Factor | Metrics | Method |
|--------|---------|--------|
| **Fundamental** | Earnings yield, book yield, ROE, operating margin, FCF yield, EPS growth, D/E (inverse), current ratio | Cross-sectional percentile rank across portfolio tickers |
| **Regime** | HMM bull-state posterior probability, VIX level signal, yield curve spread | 0.60×HMM + 0.25×VIX signal + 0.15×yield curve |
| **Relative Strength** | Stock vs sector ETF return over 52W/26W/13W | 0.25×52W rank + 0.35×26W rank + 0.40×13W rank |
| **Risk** | Sharpe ratio cross-sectional rank | Capped at 0.30 if β > 2.0 or drawdown < −30% |

## Output format

```
# Investment Report — YYYY-MM-DD

## Market Summary
[3–5 paragraph narrative: macro regime → sector rotation → top fundamental picks → risk warnings]

## Ticker Analysis
### TICKER — BUY
| Combined Score | 0.712 |
| Fundamental Score | 0.68 |
| Regime Score | 0.93 |
...
Key Insight: [DeepSeek one-liner citing dominant factor]
Key News: ...

## Top Picks Today
1. TICKER — Score: 0.712

## Notes
[model info and regime context]

## Metric Guide
[full scoring legend]
```

## Limitations

- HistGBM trained fresh daily on a rolling window — performance improves with more tickers providing cross-sectional diversity
- TipRanks scraping may be blocked by Cloudflare; yfinance analyst data is the fallback
- Fundamental scoring is cross-sectional: all tickers must be in the same run for relative ranking to be meaningful
- Macro regime detection requires at least 100 trading days of SPY/VIX data
- FinBERT covers English news only
- Not financial advice
