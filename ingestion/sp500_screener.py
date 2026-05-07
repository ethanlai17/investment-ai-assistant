import io

import pandas as pd
import requests
import yfinance as yf
from loguru import logger


class SP500Screener:
    _WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    def screen(self, top_n: int = 50) -> list[str]:
        tickers = self._fetch_sp500_tickers()
        logger.info(f"Fetched {len(tickers)} S&P 500 tickers")
        stage1 = self._momentum_filter(tickers)
        logger.info(f"Stage 1 complete: {len(stage1)} momentum candidates")
        result = self._fundamental_filter(stage1, keep=top_n)
        logger.info(f"Stage 2 complete: {len(result)} fundamental candidates")
        return result

    def _fetch_sp500_tickers(self) -> list[str]:
        resp = requests.get(self._WIKI_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        return [str(s).replace(".", "-") for s in tables[0]["Symbol"].tolist()]

    def _momentum_filter(self, tickers: list[str], keep: int = 150) -> list[str]:
        raw = yf.download(tickers, period="1y", auto_adjust=True, progress=False)
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw

        n = len(close)
        if n < 130:
            logger.warning(f"Only {n} price rows — using first {keep} tickers for stage 2")
            return tickers[:keep]

        ret_6m = (close.iloc[-1] / close.iloc[max(0, n - 126)]) - 1
        ret_12m = (close.iloc[-1] / close.iloc[max(0, n - 252)]) - 1

        # Positive 6m and 12m RS required
        valid_mask = (ret_6m > 0) & (ret_12m > 0)
        valid = valid_mask.index[valid_mask].tolist()

        if len(valid) < keep:
            valid = ret_6m.index[ret_6m > 0].tolist()
        if len(valid) < keep:
            return tickers[:keep]

        scores = pd.DataFrame({"r6": ret_6m[valid], "r12": ret_12m[valid]}).fillna(0)
        scores["score"] = (
            0.5 * scores["r6"].rank(pct=True) + 0.5 * scores["r12"].rank(pct=True)
        )
        return scores.nlargest(keep, "score").index.tolist()

    def _fundamental_filter(self, tickers: list[str], keep: int) -> list[str]:
        records = []
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                peg = info.get("pegRatio")
                records.append({
                    "ticker": ticker,
                    "roe": info.get("returnOnEquity"),
                    "peg_inv": -float(peg) if peg and float(peg) > 0 else None,
                    "eg": info.get("earningsQuarterlyGrowth"),
                })
            except Exception:
                records.append({"ticker": ticker, "roe": None, "peg_inv": None, "eg": None})

        df = pd.DataFrame(records).set_index("ticker").dropna(how="all")

        def pct_rank(s: pd.Series) -> pd.Series:
            r = s.rank(pct=True, na_option="keep")
            return r.fillna(r.median() if r.notna().any() else 0.5)

        df["score"] = (
            0.30 * pct_rank(df["roe"])
            + 0.30 * pct_rank(df["peg_inv"])
            + 0.40 * pct_rank(df["eg"])
        )

        top = df.nlargest(keep, "score").index.tolist()
        extras = [t for t in tickers if t not in top]
        return (top + extras)[:keep]
