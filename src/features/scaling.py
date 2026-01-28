"""Feature scaling for numerical columns."""
import joblib
import pandas as pd
import numpy as np
from typing import Optional, Dict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import skew


SCALERS = {
    'standard': StandardScaler,
    'minmax': MinMaxScaler,
    'robust': RobustScaler
}

METADATA_COLS = ['nutriscore_grade', 'split_group', 'product_name', 'brands', 'code']


class FeatureScaler(BaseEstimator, TransformerMixin):
    """Scales numerical features using selected method."""

    def __init__(self, method: str = 'auto', skew_threshold: float = 1.0):
        self.method = method
        self.skew_threshold = skew_threshold
        self.scalers_: Dict[str, BaseEstimator] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureScaler':
        """Fit scalers for numerical columns."""
        self.scalers_ = {}

        numerical_cols = X.select_dtypes(include=[np.number]).columns
        features_to_scale = [col for col in numerical_cols if col not in METADATA_COLS]

        if len(features_to_scale) == 0:
            raise ValueError(
                "No numerical features found to scale. "
                "Ensure input contains numerical columns (excluding metadata)."
            )

        for feature in features_to_scale:
            if self.method == 'auto':
                scaler = self._select_scaler_by_skewness(X[feature])
            else:
                scaler = SCALERS[self.method]()

            scaler.fit(X[feature].values.reshape(-1, 1))
            self.scalers_[feature] = scaler

        return self

    def _select_scaler_by_skewness(self, X_feature: pd.Series) -> BaseEstimator:
        """Pick scaler based on skewness (MinMax if high skew, else Standard)."""
        feature_skewness = abs(skew(X_feature.dropna()))
        return MinMaxScaler() if feature_skewness > self.skew_threshold else StandardScaler()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scalers to features."""
        X_scaled = X.copy()

        for feature, scaler in self.scalers_.items():
            if feature in X_scaled.columns:
                X_scaled[feature] = scaler.transform(
                    X_scaled[feature].values.reshape(-1, 1)
                ).flatten()

        return X_scaled

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit and transform."""
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        """Save to file."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'FeatureScaler':
        """Load from file."""
        return joblib.load(path)
