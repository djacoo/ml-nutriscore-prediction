import joblib
import pandas as pd
import numpy as np
from typing import Optional, Dict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import skew
from tqdm import tqdm


SCALERS = {
    'standard': StandardScaler,
    'minmax': MinMaxScaler,
    'robust': RobustScaler
}

METADATA_COLS = ['nutriscore_grade', 'split_group', 'product_name', 'brands', 'code']


class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = 'auto', skew_threshold: float = 1.0):
        self.method = method
        self.skew_threshold = skew_threshold
        self.scalers_: Dict[str, BaseEstimator] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureScaler':
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
        feature_skewness = abs(skew(X_feature.dropna()))
        return MinMaxScaler() if feature_skewness > self.skew_threshold else StandardScaler()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_scaled = X.copy()

        scalers_list = [(feature, scaler) for feature, scaler in self.scalers_.items()
                        if feature in X_scaled.columns]

        scaler_types = {}
        for _, scaler in scalers_list:
            scaler_name = type(scaler).__name__
            scaler_types[scaler_name] = scaler_types.get(scaler_name, 0) + 1

        with tqdm(total=len(scalers_list), desc="           Step 2.5: Scaling numerical features",
                  unit="feature", leave=False, mininterval=0.05, miniters=1) as pbar:
            for feature, scaler in scalers_list:
                X_scaled[feature] = scaler.transform(
                    X_scaled[feature].values.reshape(-1, 1)
                ).flatten()
                pbar.update(1)

        if scalers_list:
            print(f"                     Operation: Feature scaling")
            print(f"                              - Scaled {len(scalers_list)} numerical features")
            scaler_summary = ", ".join([f"{count} {name.replace('Scaler', '')}"
                                       for name, count in scaler_types.items()])
            print(f"                              - Methods: {scaler_summary}")
            if self.method == 'auto':
                print(f"                              - Selection: Auto (based on skewness threshold={self.skew_threshold})")

        return X_scaled

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'FeatureScaler':
        return joblib.load(path)
