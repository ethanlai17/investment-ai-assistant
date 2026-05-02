import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from loguru import logger

from analysis.features import compute_features, build_training_set, build_prediction_row
from models.types import Prediction


_MIN_TRAINING_ROWS = 30


class PricePredictor:
    def __init__(self, training_window_days: int = 60):
        self.training_window = training_window_days

    def predict(
        self,
        ticker: str,
        price_df: pd.DataFrame,
        sentiment_score: float,
    ) -> Prediction | None:
        try:
            enriched = compute_features(price_df, sentiment_score)
            X_train, y_train = build_training_set(enriched, self.training_window)

            if len(X_train) < _MIN_TRAINING_ROWS:
                logger.warning(
                    f"{ticker}: only {len(X_train)} training rows — skipping prediction"
                )
                return None

            model, scaler = self._fit(X_train, y_train)
            pred_row = build_prediction_row(price_df, sentiment_score)

            if pred_row.empty:
                logger.warning(f"{ticker}: prediction row is empty after feature computation")
                return None

            X_pred = scaler.transform(pred_row)
            proba = model.predict_proba(X_pred)[0]
            # proba[1] = P(up), proba[0] = P(down)
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
    ) -> tuple[RandomForestClassifier, StandardScaler]:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_scaled, y)
        return model, scaler

    def _confidence(self, up_probability: float) -> float:
        # Confidence reflects decisiveness: distance from 0.5
        distance = abs(up_probability - 0.5) * 2  # 0 at 0.5, 1 at 0 or 1
        return round(0.5 + distance * 0.5, 4)  # maps to [0.5, 1.0]
