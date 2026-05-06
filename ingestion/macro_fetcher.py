import yfinance as yf
import pandas as pd
from loguru import logger

from models.types import MacroData


class MacroFetcher:
    def fetch(self, lookback_days: int = 252) -> MacroData | None:
        try:
            raw = yf.download(
                ["SPY", "^VIX", "^TNX", "^IRX"],
                period=f"{lookback_days + 10}d",
                auto_adjust=True,
                progress=False,
            )
            close = raw["Close"]

            spy = close["SPY"].dropna()
            vix = close["^VIX"].dropna()
            tnx = close["^TNX"].dropna()  # 10Y yield
            irx = close["^IRX"].dropna()  # 13-week T-bill (proxy for short rate)

            spy_returns = spy.pct_change().dropna()

            # Align on common dates
            common = spy_returns.index.intersection(vix.index).intersection(tnx.index).intersection(irx.index)
            spy_returns = spy_returns.loc[common]
            vix_aligned = vix.loc[common]
            tnx_aligned = tnx.loc[common]
            irx_aligned = irx.loc[common]

            # ^IRX is annualised percent, ^TNX same; spread in percentage points
            yield_spread = tnx_aligned - irx_aligned

            return MacroData(
                spy_returns=spy_returns,
                vix_levels=vix_aligned,
                yield_spread=yield_spread,
                current_vix=float(vix_aligned.iloc[-1]),
                current_spread=float(yield_spread.iloc[-1]),
            )
        except Exception as exc:
            logger.warning(f"Macro fetch failed — {exc}")
            return None
