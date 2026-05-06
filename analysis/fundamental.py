import numpy as np
from scipy.stats import rankdata
from loguru import logger

from models.types import FundamentalData


# (field, weight, higher_is_better)
_FACTORS: list[tuple[str, float, bool]] = [
    ("earnings_yield", 0.15, True),   # 1/PE
    ("book_yield", 0.10, True),        # 1/PB
    ("roe", 0.15, True),
    ("operating_margin", 0.10, True),
    ("fcf_yield", 0.15, True),
    ("earnings_growth", 0.15, True),
    ("debt_to_equity", 0.10, False),   # lower is better
    ("current_ratio", 0.10, True),
]


def _safe(val: float | None) -> float | None:
    if val is None:
        return None
    if not np.isfinite(val):
        return None
    return float(val)


def _earnings_yield(fd: FundamentalData) -> float | None:
    if fd.pe_ratio and fd.pe_ratio > 0:
        return 1.0 / fd.pe_ratio
    return None


def _book_yield(fd: FundamentalData) -> float | None:
    if fd.pb_ratio and fd.pb_ratio > 0:
        return 1.0 / fd.pb_ratio
    return None


def _extract(fd: FundamentalData, field: str) -> float | None:
    if field == "earnings_yield":
        return _safe(_earnings_yield(fd))
    if field == "book_yield":
        return _safe(_book_yield(fd))
    return _safe(getattr(fd, field, None))


class FundamentalScorer:
    def score(self, fundamentals: dict[str, FundamentalData | None]) -> dict[str, float]:
        tickers = [t for t, fd in fundamentals.items() if fd is not None]
        if not tickers:
            return {}

        n = len(tickers)
        composite = {t: 0.0 for t in tickers}

        for field, weight, higher_is_better in _FACTORS:
            values = {t: _extract(fundamentals[t], field) for t in tickers}  # type: ignore[arg-type]
            available = [t for t in tickers if values[t] is not None]

            if not available:
                # Assign 0.5 to all (median rank)
                for t in tickers:
                    composite[t] += weight * 0.5
                continue

            raw = np.array([values[t] for t in available], dtype=float)
            ranks = rankdata(raw)  # 1..n ascending
            percentiles = (ranks - 1) / max(len(available) - 1, 1)  # 0..1

            if not higher_is_better:
                percentiles = 1.0 - percentiles

            rank_map = dict(zip(available, percentiles))
            for t in tickers:
                composite[t] += weight * rank_map.get(t, 0.5)

        result = {t: round(float(composite[t]), 4) for t in tickers}
        logger.debug(f"Fundamental scores: {result}")
        return result
