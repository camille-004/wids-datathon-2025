import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier


class XGBoostMultiTarget(BaseEstimator, ClassifierMixin):
    """XGBoost-based multi-target classifier for ADHD and sex prediction."""

    def __init__(self, random_state: int = 42, n_estimators: int = 200) -> None:
        """Initialize model."""
        self.random_state = random_state
        self.n_estimators = n_estimators

        adhd_params = {
            "n_estimators": n_estimators,
            "learning_rate": 0.1,
            "max_depth": 5,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.1,
            "random_state": random_state,
            "scale_pos_weight": 1.0,
            "tree_method": "hist",
            "eval_metric": ["auc", "logloss"],
        }

        sex_params = adhd_params.copy()

        self.adhd_classifier = XGBClassifier(**adhd_params)
        self.sex_classifier = XGBClassifier(**sex_params)

        self.adhd_threshold = 0.5
        self.sex_threshold = 0.5

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "XGBoostMultiTarget":
        """Fit the model."""
        print("\nFitting XGBoost model...")

        adhd_pos = np.sum(y["ADHD_Outcome"])
        adhd_neg = len(y["ADHD_Outcome"]) - adhd_pos
        sex_pos = np.sum(y["Sex_F"])
        sex_neg = len(y["Sex_F"]) - sex_pos

        adhd_scale = adhd_neg / adhd_pos
        sex_scale = sex_neg / sex_pos

        self.adhd_classifier.set_params(scale_pos_weight=adhd_scale)
        self.sex_classifier.set_params(scale_pos_weight=sex_scale)

        print("\nClass distributions:")
        print(f"ADHD positive: {adhd_pos}, negative: {adhd_neg}")
        print(f"Sex positive: {sex_pos}, negative: {sex_neg}")
        print(f"Scale weights - ADHD: {adhd_scale:.2f}, Sex: {sex_scale:.2f}")

        eval_set = [(X, y["ADHD_Outcome"])]

        self.adhd_classifier.fit(X, y["ADHD_Outcome"], eval_set=eval_set, verbose=False)

        self.sex_classifier.fit(
            X, y["Sex_F"], eval_set=[(X, y["Sex_F"])], verbose=False
        )

        adhd_proba = self.adhd_classifier.predict_proba(X)[:, 1]
        sex_proba = self.sex_classifier.predict_proba(X)[:, 1]

        thresholds = np.linspace(0.3, 0.7, 20)
        best_adhd_score = 0
        best_sex_score = 0

        print("\nOptimizing thresholds...")
        for threshold in thresholds:
            adhd_pred = (adhd_proba >= threshold).astype(int)
            score = f1_score(y["ADHD_Outcome"], adhd_pred)
            if score > best_adhd_score:
                best_adhd_score = score
                self.adhd_threshold = threshold

            sex_pred = (sex_proba >= threshold).astype(int)
            score = f1_score(y["Sex_F"], sex_pred)
            if score > best_sex_score:
                best_sex_score = score
                self.sex_threshold = threshold

        print(
            f"Optimal thresholds - ADHD: {self.adhd_threshold:.3f}, Sex: {self.sex_threshold:.3f}"
        )
        print(
            f"Best scores - ADHD F1: {best_adhd_score:.3f}, Sex F1: {best_sex_score:.3f}"
        )

        return self

    def predict_proba(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predict class probabilities."""
        adhd_proba = self.adhd_classifier.predict_proba(X)
        sex_proba = self.sex_classifier.predict_proba(X)
        return adhd_proba, sex_proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict classes."""
        adhd_proba, sex_proba = self.predict_proba(X)

        adhd_pred = (adhd_proba[:, 1] >= self.adhd_threshold).astype(int)
        sex_pred = (sex_proba[:, 1] >= self.sex_threshold).astype(int)

        return np.column_stack([adhd_pred, sex_pred])

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> dict[str, float]:
        """Evaluate model performance."""
        # Get predictions
        y_pred = self.predict(X)
        adhd_proba, sex_proba = self.predict_proba(X)

        # Calculate metrics
        metrics = {
            "adhd_accuracy": accuracy_score(y["ADHD_Outcome"], y_pred[:, 0]),
            "adhd_f1": f1_score(y["ADHD_Outcome"], y_pred[:, 0]),
            "adhd_auc": roc_auc_score(y["ADHD_Outcome"], adhd_proba[:, 1]),
            "sex_accuracy": accuracy_score(y["Sex_F"], y_pred[:, 1]),
            "sex_f1": f1_score(y["Sex_F"], y_pred[:, 1]),
            "sex_auc": roc_auc_score(y["Sex_F"], sex_proba[:, 1]),
        }

        # Add combined metrics
        metrics["combined_accuracy"] = (
            metrics["adhd_accuracy"] + metrics["sex_accuracy"]
        ) / 2
        metrics["combined_f1"] = (metrics["adhd_f1"] + metrics["sex_f1"]) / 2
        metrics["combined_auc"] = (metrics["adhd_auc"] + metrics["sex_auc"]) / 2

        return metrics

    def get_feature_importance(self) -> dict[str, pd.DataFrame]:
        """Get feature importance scores."""
        importance = {
            "adhd": pd.DataFrame(
                {
                    "feature": self.adhd_classifier.feature_names_in_,
                    "importance": self.adhd_classifier.feature_importances_,
                }
            ).sort_values("importance", ascending=False),
            "sex": pd.DataFrame(
                {
                    "feature": self.sex_classifier.feature_names_in_,
                    "importance": self.sex_classifier.feature_importances_,
                }
            ).sort_values("importance", ascending=False),
        }
        return importance
