import numpy as np
import pandas as pd
from scipy.stats import rankdata
from loguru import logger

from models.types import RiskMetrics


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

                raw[ticker] = {
                    "sharpe": sharpe,
                    "mdd": mdd,
                    "beta": beta,
                    "vol": ann_vol,
                }
            except Exception as exc:
                logger.warning(f"{ticker}: risk calc failed — {exc}")
                raw[ticker] = {"sharpe": 0.0, "mdd": -0.5, "beta": 1.0, "vol": 0.3}

        # Cross-sectional Sharpe percentile → risk_score
        sharpes = np.array([raw[t]["sharpe"] for t in tickers])
        if len(tickers) > 1:
            ranks = (rankdata(sharpes) - 1) / max(len(tickers) - 1, 1)
        else:
            ranks = np.array([0.5])
        sharpe_rank = dict(zip(tickers, ranks))

        results: dict[str, RiskMetrics] = {}
        for ticker in tickers:
            r = raw[ticker]
            mdd = r["mdd"]
            beta = r["beta"]
            base_score = float(sharpe_rank[ticker])
            is_high_risk = mdd < -0.30 or beta > 2.0
            risk_score = min(base_score, 0.30) if is_high_risk else base_score

            results[ticker] = RiskMetrics(
                ticker=ticker,
                sharpe_ratio=round(r["sharpe"], 4),
                max_drawdown=round(mdd, 4),
                beta=round(beta, 4),
                annualised_vol=round(r["vol"], 4),
                risk_score=round(risk_score, 4),
                is_high_risk=is_high_risk,
            )
            logger.debug(
                f"{ticker}: sharpe={r['sharpe']:.3f}, mdd={mdd:.3f}, "
                f"beta={beta:.3f}, risk_score={risk_score:.4f}, high_risk={is_high_risk}"
            )

        return results
