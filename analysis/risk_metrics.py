import numpy as np
import pandas as pd
from scipy.stats import rankdata
from loguru import logger

from models.types import RiskMetrics


def _pct_rank_in_ref(value: float, ref: np.ndarray) -> float:
    if len(ref) == 0:
        return 0.5
    return int(np.searchsorted(ref, value)) / len(ref)


def _max_drawdown(prices: pd.Series) -> float:
    peak = prices.cummax()
    drawdown = (prices - peak) / peak
    return float(drawdown.min())


def _beta(stock_returns: pd.Series, spy_returns: pd.Series) -> float:
    common = stock_returns.index.intersection(spy_returns.index)
    if len(common) < 30:
        return 1.0
    s = stock_returns.loc[common].values
    m = spy_returns.loc[common].values
    cov = np.cov(s, m)
    var_market = cov[1, 1]
    if var_market == 0:
        return 1.0
    return float(cov[0, 1] / var_market)


class RiskCalculator:
    def compute_all(
        self,
        price_data: dict[str, pd.DataFrame],
        spy_returns: pd.Series,
        risk_free_rate: float,
        reference_sharpes: np.ndarray | None = None,
        reference_sortinos: np.ndarray | None = None,
    ) -> dict[str, RiskMetrics]:
        tickers = list(price_data.keys())
        raw: dict[str, dict] = {}

        for ticker in tickers:
            try:
                close = price_data[ticker]["Close"].squeeze().dropna()
                rets = close.pct_change().dropna()

                ann_return = float(rets.mean() * 252)
                ann_vol = float(rets.std() * np.sqrt(252))
                sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0
                mdd = _max_drawdown(close)
                beta = _beta(rets, spy_returns)

                threshold = np.percentile(rets.values, 5)
                tail = rets[rets <= threshold]
                cvar_95 = float(tail.mean()) if len(tail) > 0 else float(threshold)

                downside = rets[rets < 0]
                downside_dev = float(downside.std() * np.sqrt(252)) if len(downside) > 1 else ann_vol
                sortino = (ann_return - risk_free_rate) / downside_dev if downside_dev > 0 else 0.0

                raw[ticker] = {
                    "sharpe": sharpe,
                    "mdd": mdd,
                    "beta": beta,
                    "vol": ann_vol,
                    "cvar_95": cvar_95,
                    "sortino": sortino,
                }
            except Exception as exc:
                logger.warning(f"{ticker}: risk calc failed — {exc}")
                raw[ticker] = {"sharpe": 0.0, "mdd": -0.5, "beta": 1.0, "vol": 0.3, "cvar_95": -0.03, "sortino": 0.0}

        if reference_sharpes is not None and len(reference_sharpes) > 0:
            sharpe_rank = {
                t: _pct_rank_in_ref(raw[t]["sharpe"], reference_sharpes) for t in tickers
            }
        else:
            sharpes = np.array([raw[t]["sharpe"] for t in tickers])
            if len(tickers) > 1:
                ranks = (rankdata(sharpes) - 1) / max(len(tickers) - 1, 1)
            else:
                ranks = np.array([0.5])
            sharpe_rank = dict(zip(tickers, ranks))

        if reference_sortinos is not None and len(reference_sortinos) > 0:
            sortino_rank = {
                t: _pct_rank_in_ref(raw[t]["sortino"], reference_sortinos) for t in tickers
            }
        else:
            sortinos = np.array([raw[t]["sortino"] for t in tickers])
            if len(tickers) > 1:
                sortino_ranks_arr = (rankdata(sortinos) - 1) / max(len(tickers) - 1, 1)
            else:
                sortino_ranks_arr = np.array([0.5])
            sortino_rank = dict(zip(tickers, sortino_ranks_arr))

        results: dict[str, RiskMetrics] = {}
        for ticker in tickers:
            r = raw[ticker]
            mdd = r["mdd"]
            beta = r["beta"]
            cvar = r["cvar_95"]

            is_high_risk = cvar < -0.05 or beta > 2.0

            blended = 0.60 * float(sortino_rank[ticker]) + 0.40 * float(sharpe_rank[ticker])
            cvar_penalty = min(0.20, max(0.0, (-cvar - 0.01) / 0.04 * 0.20))
            risk_score = max(0.0, blended - cvar_penalty)

            results[ticker] = RiskMetrics(
                ticker=ticker,
                sharpe_ratio=round(r["sharpe"], 4),
                max_drawdown=round(mdd, 4),
                beta=round(beta, 4),
                annualised_vol=round(r["vol"], 4),
                cvar_95=round(cvar, 5),
                sortino_ratio=round(r["sortino"], 4),
                risk_score=round(risk_score, 4),
                is_high_risk=is_high_risk,
            )
            logger.debug(
                f"{ticker}: sharpe={r['sharpe']:.3f}, sortino={r['sortino']:.3f}, "
                f"cvar95={cvar:.4f}, mdd={mdd:.3f}, beta={beta:.3f}, "
                f"risk_score={risk_score:.4f}, high_risk={is_high_risk}"
            )

        return results
