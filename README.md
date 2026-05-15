# Investment AI Assistant

A production-ready AI-powered investment report system with two modes: a **daily watchlist report** for a predefined set of tickers, and an **S&P 500 scanner** that screens all 503 constituents and surfaces the top 5 picks by combined score. Implements institutional-grade multi-factor analysis: Carhart 4-factor alpha, CVaR/Sortino risk scoring, Fama-French fundamental ranking, HMM regime detection, sector-relative strength, FinBERT sentiment, and XGBoost price prediction.

## What it does

### Daily watchlist report (`--run-now`)

1. **Fetches news** from Yahoo Finance RSS (yfinance fallback)
2. **Fetches analyst data** — consensus and price targets from TipRanks (yfinance fallback)
3. **Fetches fundamentals** — P/E, P/B, ROE, FCF yield, earnings growth, D/E, current ratio from yfinance
4. **Fetches macro data** — SPY returns, VIX, 10Y/2Y yield curve for regime detection
5. **Fetches sector ETF data** — maps each ticker to its GICS sector ETF (XLK, XLF, XLV…)
6. **Scores sentiment** using FinBERT (ProsusAI/finbert, 0.94 F1) — finance-specific BERT, runs locally
7. **Detects macro regime** using a 2-state Gaussian Hidden Markov Model on SPY returns + VIX + yield curve; `regime_score = 0.60×HMM + 0.25×VIX + 0.15×yield_curve`
8. **Scores fundamentals** using a Fama-French-style 8-factor cross-sectional percentile rank: earnings yield, book yield, ROE, operating margin, FCF yield, EPS growth, D/E (inverse), current ratio
9. **Measures sector-relative strength** vs sector ETF over 13/26/52-week windows; weighted 40/35/25
10. **Computes risk metrics** — Sharpe ratio, Sortino ratio (downside deviation only), CVaR 95% (tail loss), max drawdown, beta. `risk_score = 0.60×Sortino_rank + 0.40×Sharpe_rank − CVaR_penalty`. High-risk flag (CVaR < −5% daily or beta > 2.0) caps the signal at HOLD and overrides the risk label to "high risk (CVaR/beta breach)" regardless of the relative score
11. **Decomposes factor alpha** using Carhart 4-factor OLS regression (MKT-RF, SMB, HML, UMD) against Fama-French daily factors from Kenneth French's data library. Cross-sectional alpha rank = performance unexplained by market, size, value, and momentum factors
12. **Predicts 20-day forward price direction** using HistGradientBoostingClassifier on a 252-day rolling window; 15 features: RSI, MACD, Bollinger %, volume ratio, 5/20-day momentum, daily return, 52-week high proximity (George & Hwang, 2004), plus composite scores
13. **Scores with a 9-factor composite** weighted for long-term investing (see table below)
14. **Generates** per-ticker insights (DeepSeek Flash) and a macro-led market narrative (DeepSeek Pro)
15. **Emails** the report daily at 9:30 AM UK time

### S&P 500 scanner (`--scan`)

Screens all 503 S&P 500 constituents with a two-stage pre-filter, runs the full 17-step pipeline on the top 50 candidates, and returns the **top 5 picks** prioritised by signal: BUY tickers ranked by combined score first, then HOLD tickers by combined score to fill remaining slots. AVOID tickers are never included.

**Stage 1 — Momentum screen (503 → 150):** Requires positive 6-month AND 12-month returns; ranks by combined 6m/12m relative strength.

**Stage 2 — Fundamental screen (150 → 50):** Cross-sectional rank of ROE (30%), PEG inverted (30%), quarterly earnings growth (40%).

The top 50 run through the full pipeline above. Output saved to `outputs/YYYY-MM-DD-sp500-scan.{md,txt}` and emailed as "S&P 500 Top 5 Picks".

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
| `RECIPIENT_EMAIL` | Where to send the report (defaults to `SMTP_USER`) |
| `TICKERS` | Comma-separated ticker symbols, e.g. `AAPL,MSFT,NVDA` |

**Gmail App Password:** Enable 2FA → Google Account → Security → App passwords → Mail.

### 4. Commands

| Command | Description |
|---------|-------------|
| `./run.sh --run-now` | Run daily watchlist report now |
| `./run.sh --scan` | Run S&P 500 top 5 scan (~3–5 min) |
| `LOG_LEVEL=DEBUG ./run.sh --run-now` | Run with debug logging |
| `launchctl load ~/Library/LaunchAgents/com.ethanlai.investment-assistant.plist` | Enable daily scheduler at 09:30 (survives reboots) |
| `launchctl unload ~/Library/LaunchAgents/com.ethanlai.investment-assistant.plist` | Disable scheduler |
| `launchctl start com.ethanlai.investment-assistant` | Trigger scheduler run immediately |
| `launchctl print gui/$(id -u)/com.ethanlai.investment-assistant` | Check scheduler status |
| `tail -f logs/launchd.log` | Stream scheduler logs |

## Configuration reference

| Variable | Default | Description |
|----------|---------|-------------|
| `TICKERS` | *(required)* | Comma-separated tickers to analyse |
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
Wikipedia S&P 500  ──► SP500Screener ──► top 50 candidates   (--scan only)

Yahoo Finance RSS  ──► NewsFetcher
yfinance news      ──► (fallback)            ┐
yfinance OHLCV     ──► PriceFetcher          │
TipRanks/yfinance  ──► AnalystFetcher        │
yfinance info      ──► FundamentalFetcher    │  ingestion/
SPY/VIX/yields     ──► MacroFetcher          │
Sector ETFs        ──► SectorETFFetcher      ┘

                   Processing (clean, dedup, map tickers)

                   FinBERT           ──► sentiment_score       ┐
                   HMM (2-state)     ──► regime_score          │
                   Fama-French rank  ──► fundamental_score     │
                   RS vs sector ETF  ──► rs_score              │  analysis/
                   Sortino/CVaR      ──► risk_score            │
                   Carhart 4-factor  ──► carhart_alpha         │
                   HistGBM (252-day) ──► 20-day prediction     ┘

                   9-factor Recommender ──► BUY / HOLD / AVOID

                   DeepSeek Flash ──► per-ticker insights
                   DeepSeek Pro   ──► macro-led market narrative

                   Jinja2 templates ──► .md + .txt report
                   Gmail SMTP_SSL   ──► email delivery
```

## 9-Factor Recommendation Scoring

| Factor | With analyst data | Without analyst data |
|--------|-------------------|---------------------|
| ML 20-day prediction (HistGBM) | 18% | 22% |
| Fundamental score (Fama-French 8-factor) | 16% | 20% |
| Carhart alpha (4-factor OLS) | 12% | 15% |
| Macro regime (HMM + VIX + yield curve) | 12% | 15% |
| Sector-relative strength (13/26/52W) | 10% | 12% |
| Sentiment (FinBERT) | 10% | 5% |
| Risk score (Sortino/CVaR-weighted) | 9% | 11% |
| Analyst consensus (TipRanks/yfinance) | 7% | — |
| Price target upside (analyst target vs price) | 6% | — |

| Signal | Condition |
|--------|-----------|
| **BUY** | combined ≥ 0.65 AND model confidence ≥ 0.60 AND not high-risk |
| **HOLD** | combined ≥ 0.45 (or high-risk stock regardless of score) |
| **AVOID** | combined < 0.45 |

High-risk = CVaR 95% < −5% daily OR beta > 2.0.

## Factor Definitions

| Factor | Theory / Model | Method |
|--------|---------------|--------|
| **Fundamental** | Fama-French (1993, 2015) | 8-factor cross-sectional percentile rank across portfolio tickers |
| **Regime** | Hamilton HMM (1989) | 2-state Gaussian HMM on SPY returns; `0.60×bull_prob + 0.25×VIX_signal + 0.15×yield_curve` |
| **Relative Strength** | Jegadeesh & Titman (1993) momentum | Stock vs sector ETF return; `0.40×13W + 0.35×26W + 0.25×52W` cross-sectional rank |
| **Risk Score** | Sharpe (1966), Sortino & van der Meer (1991), CVaR (Artzner et al. 1999) | `0.60×Sortino_rank + 0.40×Sharpe_rank − CVaR_penalty`; penalty scales linearly from CVaR −1% to −5% daily |
| **Carhart Alpha** | Carhart (1997) 4-factor model | OLS regression vs MKT-RF, SMB, HML, UMD; annualised intercept cross-sectionally ranked |
| **ML Prediction** | George & Hwang (2004) 52W high + technical features | HistGBM, 15 features, 252-day rolling window, 20-day forward binary target |
| **PT Upside** | Analyst target premium | `(target − price) / price` clipped to [−50%, +100%], normalised to [0, 1] |

## Limitations

- HistGBM trained fresh daily — cross-sectional diversity improves with more tickers in the same run
- TipRanks scraping may be blocked by Cloudflare; yfinance analyst data is the fallback
- Fama-French factor data has a 1–2 business day lag from Kenneth French's library; regression uses the overlapping window automatically
- Fundamental scoring is cross-sectional: all tickers must be in the same run for relative ranking to be meaningful
- Macro regime detection requires at least 100 trading days of SPY/VIX data
- FinBERT covers English news only
- Not financial advice
