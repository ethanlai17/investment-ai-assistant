import numpy as np
import pandas as pd
from scipy.stats import rankdata
from loguru import logger

from models.types import RelativeStrength


def _pct_rank_in_ref(value: float, ref: np.ndarray) -> float:
    if len(ref) == 0:
        return 0.5
    return int(np.searchsorted(ref, value)) / len(ref)


def _cumulative_return(price_series: pd.Series, days: int) -> float | None:
    s = price_series.dropna()
    if len(s) < days + 1:
        return None
    start = float(s.iloc[-(days + 1)])
    end = float(s.iloc[-1])
    if start == 0:
        return None
    return (end - start) / start


def _rs_ratio(stock_ret: float | None, etf_ret: float | None) -> float | None:
    if stock_ret is None or etf_ret is None:
        return None
    # Avoid divide-by-zero; treat ETF return of 0 as very small positive
    denom = etf_ret if etf_ret != 0 else 1e-6
    return stock_ret / denom


class RelativeStrengthCalculator:
    def compute_all(
        self,
        price_data: dict[str, pd.DataFrame],
        ticker_etf: dict[str, str],
        ticker_sector: dict[str, str],
        etf_prices: dict[str, pd.DataFrame],
        reference: dict[str, np.ndarray] | None = None,
    ) -> dict[str, RelativeStrength]:
        tickers = list(price_data.keys())
        raw_rs: dict[str, dict[str, float | None]] = {}

        for ticker in tickers:
            etf = ticker_etf.get(ticker, "SPY")
            price_series = price_data[ticker]["Close"].squeeze()
            etf_series = etf_prices.get(etf, pd.DataFrame())
            etf_close = etf_series["Close"].squeeze() if not etf_series.empty else pd.Series(dtype=float)

            rs52 = _rs_ratio(
                _cumulative_return(price_series, 252),
                _cumulative_return(etf_close, 252),
            )
            rs26 = _rs_ratio(
                _cumulative_return(price_series, 126),
                _cumulative_return(etf_close, 126),
            )
            rs13 = _rs_ratio(
                _cumulative_return(price_series, 63),
                _cumulative_return(etf_close, 63),
            )
            raw_rs[ticker] = {"rs_52w": rs52, "rs_26w": rs26, "rs_13w": rs13}

        if reference is not None:
            # Rank each ticker's raw RS ratio against the S&P 500 reference distribution
            # so both pipelines produce identical scores for the same stock.
            rank52 = {
                t: _pct_rank_in_ref(raw_rs[t]["rs_52w"] or 1.0, reference["rs_52w"])
                for t in tickers
            }
            rank26 = {
                t: _pct_rank_in_ref(raw_rs[t]["rs_26w"] or 1.0, reference["rs_26w"])
                for t in tickers
            }
            rank13 = {
                t: _pct_rank_in_ref(raw_rs[t]["rs_13w"] or 1.0, reference["rs_13w"])
                for t in tickers
            }
        else:
            # Fallback: cross-sectional rank within current batch only
            def _rank(key: str) -> dict[str, float]:
                vals = {t: raw_rs[t][key] for t in tickers if raw_rs[t][key] is not None}
                if not vals:
                    return {t: 0.5 for t in tickers}
                arr = np.array(list(vals.values()))
                ranks = (rankdata(arr) - 1) / max(len(arr) - 1, 1)
                rank_map = dict(zip(vals.keys(), ranks))
                return {t: float(rank_map.get(t, 0.5)) for t in tickers}

            rank52 = _rank("rs_52w")
            rank26 = _rank("rs_26w")
            rank13 = _rank("rs_13w")

        results: dict[str, RelativeStrength] = {}
        for ticker in tickers:
            rs_score = round(
                0.25 * rank52[ticker] + 0.35 * rank26[ticker] + 0.40 * rank13[ticker], 4
            )
            etf = ticker_etf.get(ticker, "SPY")
            sector = ticker_sector.get(ticker, "Unknown")
            rr = raw_rs[ticker]
            results[ticker] = RelativeStrength(
                ticker=ticker,
                rs_52w=round(rr["rs_52w"] or 0.0, 4),
                rs_26w=round(rr["rs_26w"] or 0.0, 4),
                rs_13w=round(rr["rs_13w"] or 0.0, 4),
                rs_score=rs_score,
                sector=sector,
                sector_etf=etf,
            )
            logger.debug(f"{ticker}: rs_score={rs_score:.4f} ({sector}/{etf})")

        return results
