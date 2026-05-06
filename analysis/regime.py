import numpy as np
from hmmlearn.hmm import GaussianHMM
from loguru import logger

from models.types import MacroData, RegimeState


def _vix_signal(vix: float) -> float:
    if vix < 20:
        return 1.0
    if vix <= 30:
        return 0.5
    return 0.0


def _yield_curve_signal(spread: float) -> float:
    # spread in percentage points; +2 = steep normal, -1 = deeply inverted
    # Map [-1.5, +2.5] linearly to [0, 1]
    clipped = max(-1.5, min(2.5, spread))
    return round((clipped + 1.5) / 4.0, 4)


class RegimeDetector:
    def detect(self, macro: MacroData) -> RegimeState:
        try:
            returns = macro.spy_returns.values.reshape(-1, 1)
            vix = macro.vix_levels.values.reshape(-1, 1)

            # Normalise both features before fitting
            r_std = returns.std() or 1e-8
            v_std = vix.std() or 1e-8
            X = np.hstack([returns / r_std, vix / v_std])

            model = GaussianHMM(
                n_components=2,
                covariance_type="diag",
                n_iter=100,
                random_state=42,
            )
            model.fit(X)

            # Identify which state is "bull" by higher mean SPY return
            state_means = model.means_[:, 0]  # first feature is returns
            bull_state = int(np.argmax(state_means))

            posteriors = model.predict_proba(X)
            bull_probability = float(posteriors[-1, bull_state])

            current_state = "bull" if bull_probability >= 0.5 else "bear"
            vix_sig = _vix_signal(macro.current_vix)
            yc_sig = _yield_curve_signal(macro.current_spread)

            regime_score = round(
                0.60 * bull_probability + 0.25 * vix_sig + 0.15 * yc_sig, 4
            )

            logger.debug(
                f"Regime: {current_state}, bull_prob={bull_probability:.3f}, "
                f"VIX={macro.current_vix:.1f}, spread={macro.current_spread:.2f}, "
                f"regime_score={regime_score:.4f}"
            )

            return RegimeState(
                current_state=current_state,
                regime_score=regime_score,
                bull_probability=round(bull_probability, 4),
                vix_signal=vix_sig,
                yield_curve_signal=yc_sig,
            )
        except Exception as exc:
            logger.warning(f"Regime detection failed — {exc}")
            return RegimeState(
                current_state="unknown",
                regime_score=0.5,
                bull_probability=0.5,
                vix_signal=0.5,
                yield_curve_signal=0.5,
            )
