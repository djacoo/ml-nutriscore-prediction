"""Categorical feature encoding."""
import joblib
import pandas as pd
import numpy as np
from collections import Counter
from typing import Optional, Dict, List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, MultiLabelBinarizer


CATEGORICAL_FEATURES = ["countries", "pnns_groups_1", "pnns_groups_2"]


class FeatureEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical features using different strategies per feature."""

    def __init__(self, top_n_countries: int = 15):
        self.top_n_countries = top_n_countries
        self.encoders_: Dict[str, BaseEstimator] = {}
        self.top_countries_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEncoder':
        """Fit encoders for each categorical feature."""
        self.encoders_ = {}
        self.top_countries_ = None

        for feature in CATEGORICAL_FEATURES:
            if feature not in X.columns:
                continue

            if feature == "countries":
                self._fit_countries_encoder(X[feature])
            elif feature == "pnns_groups_1":
                self._fit_onehot_encoder(X[feature])
            elif feature == "pnns_groups_2":
                self._fit_target_encoder(X[feature], y)

        return self

    def _fit_countries_encoder(self, X_countries: pd.Series) -> None:
        """Fit MultiLabelBinarizer on top N countries."""
        country_lists = X_countries.apply(
            lambda x: [c.strip() for c in str(x).split(',')
                      if c.strip() and c.strip() != 'unknown']
        ).tolist()

        country_counts = Counter()
        for country_list in country_lists:
            country_counts.update(country_list)

        self.top_countries_ = [
            country for country, _ in country_counts.most_common(self.top_n_countries)
        ]

        filtered_data = [
            [c for c in country_list if c in self.top_countries_]
            for country_list in country_lists
        ]

        encoder = MultiLabelBinarizer(classes=self.top_countries_)
        encoder.fit(filtered_data)
        self.encoders_["countries"] = encoder

    def _fit_onehot_encoder(self, X_feature: pd.Series) -> None:
        """Fit OneHotEncoder with unknown handling."""
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoder.fit(X_feature.values.reshape(-1, 1))
        self.encoders_["pnns_groups_1"] = encoder

    def _fit_target_encoder(self, X_feature: pd.Series, y: Optional[pd.Series]) -> None:
        """Fit TargetEncoder (needs target variable)."""
        if y is None:
            raise ValueError("TargetEncoder for pnns_groups_2 requires y parameter")

        encoder = TargetEncoder(smooth="auto")
        encoder.fit(X_feature.values.reshape(-1, 1), y)
        self.encoders_["pnns_groups_2"] = encoder

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted encoders to features."""
        X_encoded = X.copy()

        for feature in CATEGORICAL_FEATURES:
            if feature not in X_encoded.columns or feature not in self.encoders_:
                continue

            encoder = self.encoders_[feature]

            if isinstance(encoder, MultiLabelBinarizer):
                X_encoded = self._transform_multilabel(X_encoded, feature, encoder)
            elif isinstance(encoder, OneHotEncoder):
                X_encoded = self._transform_onehot(X_encoded, feature, encoder)
            else:
                X_encoded = self._transform_target(X_encoded, feature, encoder)

        return X_encoded

    def _transform_multilabel(
        self, X: pd.DataFrame, feature: str, encoder: MultiLabelBinarizer
    ) -> pd.DataFrame:
        """Transform multi-label feature."""
        country_lists = X[feature].apply(
            lambda x: [c.strip() for c in str(x).split(',')
                      if c.strip() and c.strip() != 'unknown']
        ).tolist()

        filtered_data = [
            [c for c in country_list if c in self.top_countries_]
            for country_list in country_lists
        ]

        transformed = encoder.transform(filtered_data)
        feature_names = [f"{feature}_{country}" for country in encoder.classes_]

        X_encoded = X.drop(columns=[feature])
        X_encoded = pd.concat([
            X_encoded,
            pd.DataFrame(transformed, columns=feature_names, index=X.index)
        ], axis=1)

        return X_encoded

    def _transform_onehot(
        self, X: pd.DataFrame, feature: str, encoder: OneHotEncoder
    ) -> pd.DataFrame:
        """Transform with OneHotEncoder."""
        transformed = encoder.transform(X[feature].values.reshape(-1, 1))
        feature_names = encoder.get_feature_names_out([feature])

        X_encoded = X.drop(columns=[feature])
        X_encoded = pd.concat([
            X_encoded,
            pd.DataFrame(transformed, columns=feature_names, index=X.index)
        ], axis=1)

        return X_encoded

    def _transform_target(
        self, X: pd.DataFrame, feature: str, encoder: TargetEncoder
    ) -> pd.DataFrame:
        """Transform with TargetEncoder."""
        transformed = encoder.transform(X[feature].values.reshape(-1, 1))
        transformed = np.asarray(transformed).squeeze()
        if transformed.ndim > 1:
            transformed = transformed[:, 0]

        X[feature] = transformed
        return X

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit and transform."""
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        """Save to file."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'FeatureEncoder':
        """Load from file."""
        return joblib.load(path)
