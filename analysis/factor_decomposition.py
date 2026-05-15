import io
import zipfile
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
from scipy.stats import rankdata
from loguru import logger

from models.types import FactorExposure

_FF3_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
_MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _read_french_csv(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            raw = f.read().decode("latin-1")
    # Skip header lines until we hit the data block (first line starting with a digit)
    lines = raw.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() and l.strip()[0].isdigit())
    # Find end: next blank or non-numeric block after start
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].strip() and not lines[i].strip()[0].isdigit():
            end = i
            break
    data_str = "\n".join(lines[start:end])
    df = pd.read_csv(io.StringIO(data_str), header=None)
    return df


def _fetch_ff4_factors(start: str) -> pd.DataFrame | None:
    try:
        r3 = requests.get(_FF3_URL, headers=_HEADERS, timeout=20)
        rm = requests.get(_MOM_URL, headers=_HEADERS, timeout=20)
        r3.raise_for_status()
        rm.raise_for_status()

        ff3 = _read_french_csv(r3.content)
        ff3.columns = ["date", "Mkt-RF", "SMB", "HML", "RF"]
        ff3["date"] = pd.to_datetime(ff3["date"], format="%Y%m%d", errors="coerce")
        ff3 = ff3.dropna(subset=["date"]).set_index("date")

        mom = _read_french_csv(rm.content)
        mom.columns = ["date", "Mom"]
        mom["date"] = pd.to_datetime(mom["date"], format="%Y%m%d", errors="coerce")
        mom = mom.dropna(subset=["date"]).set_index("date")

        merged = ff3.join(mom, how="inner") / 100.0
        merged = merged[merged.index >= pd.Timestamp(start)]
        return merged
    except Exception as exc:
        logger.warning(f"Fama-French 4-factor fetch failed — {exc}")
        return None


def _run_regression(stock_rets: pd.Series, ff_factors: pd.DataFrame) -> dict | None:
    common = stock_rets.index.intersection(ff_factors.index)
    if len(common) < 60:
        return None
    rf = ff_factors.loc[common, "RF"].values
    y = stock_rets.loc[common].values - rf
    X = ff_factors.loc[common, ["Mkt-RF", "SMB", "HML", "Mom"]].values
    X_const = np.column_stack([np.ones(len(X)), X])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X_const, y, rcond=None)
        y_pred = X_const @ coeffs
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        alpha_annual = (1 + float(coeffs[0])) ** 252 - 1
        return {
            "alpha": alpha_annual,
            "beta_mkt": float(coeffs[1]),
            "beta_smb": float(coeffs[2]),
            "beta_hml": float(coeffs[3]),
            "beta_umd": float(coeffs[4]),
            "r_squared": r2,
        }
    except Exception:
        return None


class FactorDecomposer:
    def compute_all(
        self,
        price_data: dict[str, pd.DataFrame],
        lookback_days: int = 400,
    ) -> dict[str, FactorExposure]:
        start = (date.today() - timedelta(days=lookback_days)).isoformat()
        ff_factors = _fetch_ff4_factors(start)

        tickers = list(price_data.keys())
        raw: dict[str, dict | None] = {}

        for ticker in tickers:
            try:
                close = price_data[ticker]["Close"].squeeze().dropna()
                rets = close.pct_change().dropna()
                raw[ticker] = _run_regression(rets, ff_factors) if ff_factors is not None else None
            except Exception as exc:
                logger.warning(f"{ticker}: factor decomposition failed — {exc}")
                raw[ticker] = None

        available = {t: raw[t] for t in tickers if raw[t] is not None}
        if len(available) > 1:
            alphas = np.array([available[t]["alpha"] for t in available])
            ranks = (rankdata(alphas) - 1) / max(len(alphas) - 1, 1)
            alpha_norm = dict(zip(available.keys(), ranks.tolist()))
        elif len(available) == 1:
            alpha_norm = {next(iter(available)): 0.5}
        else:
            alpha_norm = {}

        results: dict[str, FactorExposure] = {}
        for ticker in tickers:
            r = raw.get(ticker)
            if r is None:
                results[ticker] = FactorExposure(
                    ticker=ticker, alpha=0.0, beta_mkt=1.0,
                    beta_smb=0.0, beta_hml=0.0, beta_umd=0.0,
                    r_squared=0.0, carhart_alpha=0.5,
                )
            else:
                results[ticker] = FactorExposure(
                    ticker=ticker,
                    alpha=round(r["alpha"], 6),
                    beta_mkt=round(r["beta_mkt"], 4),
                    beta_smb=round(r["beta_smb"], 4),
                    beta_hml=round(r["beta_hml"], 4),
                    beta_umd=round(r["beta_umd"], 4),
                    r_squared=round(r["r_squared"], 4),
                    carhart_alpha=round(alpha_norm.get(ticker, 0.5), 4),
                )
            logger.debug(
                f"{ticker}: alpha={results[ticker].alpha:.4f}, "
                f"carhart_alpha_norm={results[ticker].carhart_alpha:.4f}, "
                f"r2={results[ticker].r_squared:.4f}"
            )

        return results
