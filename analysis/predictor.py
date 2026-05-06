import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from loguru import logger

from analysis.features import compute_features, build_training_set, build_prediction_row
from models.types import Prediction


_MIN_TRAINING_ROWS = 30
_FORWARD_DAYS = 20  # predict 20-day forward direction for long-term signal


class PricePredictor:
    def __init__(self, training_window_days: int = 252):
        self.training_window = training_window_days

    def predict(
        self,
        ticker: str,
        price_df: pd.DataFrame,
        sentiment_score: float,
        fundamental_score: float = 0.5,
        regime_score: float = 0.5,
        rs_score: float = 0.5,
        risk_score: float = 0.5,
    ) -> Prediction | None:
        try:
            enriched = compute_features(
                price_df, sentiment_score, fundamental_score, regime_score, rs_score, risk_score
            )
            X_train, y_train = build_training_set(enriched, self.training_window, _FORWARD_DAYS)

            if len(X_train) < _MIN_TRAINING_ROWS:
                logger.warning(
                    f"{ticker}: only {len(X_train)} training rows — skipping prediction"
                )
                return None

            model, scaler = self._fit(X_train, y_train)
            pred_row = build_prediction_row(
                price_df, sentiment_score, fundamental_score, regime_score, rs_score, risk_score
            )

            if pred_row.empty:
                logger.warning(f"{ticker}: prediction row is empty after feature computation")
                return None

            X_pred = scaler.transform(pred_row)
            proba = model.predict_proba(X_pred)[0]
            up_prob = float(proba[1])
            confidence = self._confidence(up_prob)

            feature_snapshot = dict(zip(pred_row.columns, pred_row.iloc[0].tolist()))

            logger.debug(
                f"{ticker}: up_prob={up_prob:.3f}, confidence={confidence:.3f}, "
                f"training_rows={len(X_train)}"
            )
            return Prediction(
                ticker=ticker,
                up_probability=up_prob,
                confidence=confidence,
                feature_snapshot={k: round(v, 4) for k, v in feature_snapshot.items()},
            )
        except Exception as exc:
            logger.warning(f"{ticker}: prediction failed — {exc}")
            return None

    def _fit(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[HistGradientBoostingClassifier, StandardScaler]:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=4,
            learning_rate=0.05,
            max_bins=255,
            random_state=42,
        )
        model.fit(X_scaled, y)
        return model, scaler

    def _confidence(self, up_probability: float) -> float:
        distance = abs(up_probability - 0.5) * 2
        return round(0.5 + distance * 0.5, 4)
