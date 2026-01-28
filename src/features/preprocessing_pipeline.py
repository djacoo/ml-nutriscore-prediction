"""Complete preprocessing pipeline combining all transformers."""
import pandas as pd
import numpy as np
from typing import Optional, List
from sklearn.pipeline import Pipeline

from features.outlier_removal import MissingValueTransformer, OutlierRemovalTransformer
from features.encoding import FeatureEncoder
from features.feature_engineering import FeatureEngineer
from features.scaling import FeatureScaler
from features.dimensionality_reduction import FeatureReducer


def create_preprocessing_pipeline(
    missing_threshold: float = 0.95,
    top_n_countries: int = 15,
    scaling_method: str = 'auto',
    scaling_skew_threshold: float = 1.0,
    pca_variance_threshold: float = 0.95,
    target_col: str = 'nutriscore_grade',
    include_feature_engineering: bool = True,
    feature_engineering_kwargs: Optional[dict] = None,
    remove_statistical_outliers: bool = False,
    include_pca: bool = True
) -> Pipeline:
    """Create preprocessing pipeline with configurable steps."""
    steps = [
        ('missing_values', MissingValueTransformer(
            threshold_drop_feature=missing_threshold,
            target_col=target_col
        )),
        ('outlier_removal', OutlierRemovalTransformer(
            target_col=target_col,
            remove_statistical_outliers=remove_statistical_outliers
        )),
        ('encoding', FeatureEncoder(top_n_countries=top_n_countries)),
    ]

    if include_feature_engineering:
        fe_kwargs = feature_engineering_kwargs or {}
        steps.append(('feature_engineering', FeatureEngineer(**fe_kwargs)))

    steps.append(('scaling', FeatureScaler(method=scaling_method, skew_threshold=scaling_skew_threshold)))

    if include_pca:
        steps.append(('pca', FeatureReducer(variance_threshold=pca_variance_threshold)))

    return Pipeline(steps)


class PreprocessingPipeline:
    """High-level preprocessing pipeline wrapper with metadata handling."""

    def __init__(
        self,
        missing_threshold: float = 0.95,
        top_n_countries: int = 15,
        scaling_method: str = 'auto',
        scaling_skew_threshold: float = 1.0,
        pca_variance_threshold: float = 0.95,
        target_col: str = 'nutriscore_grade',
        split_group_col: str = 'split_group',
        preserve_cols: Optional[List[str]] = None,
        include_feature_engineering: bool = True,
        feature_engineering_kwargs: Optional[dict] = None,
        remove_statistical_outliers: bool = False,
        include_pca: bool = True
    ):
        self.target_col = target_col
        self.split_group_col = split_group_col
        self.preserve_cols = preserve_cols or []

        self.pipeline = create_preprocessing_pipeline(
            missing_threshold=missing_threshold,
            top_n_countries=top_n_countries,
            scaling_method=scaling_method,
            scaling_skew_threshold=scaling_skew_threshold,
            pca_variance_threshold=pca_variance_threshold,
            target_col=target_col,
            include_feature_engineering=include_feature_engineering,
            feature_engineering_kwargs=feature_engineering_kwargs,
            remove_statistical_outliers=remove_statistical_outliers,
            include_pca=include_pca
        )

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PreprocessingPipeline':
        """Fit pipeline on training data only to prevent data leakage."""
        if self.split_group_col in X.columns:
            train_mask = X[self.split_group_col] == 'train'
            X_train = X[train_mask]

            if y is not None:
                y_train = y[train_mask]
            elif self.target_col in X_train.columns:
                y_train = X_train[self.target_col]
            else:
                y_train = None

            print(f"Fitting on train split only: {len(X_train):,} samples")
            self.pipeline.fit(X_train, y_train)
        else:
            self.pipeline.fit(X, y)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted pipeline."""
        return self.pipeline.transform(X)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform."""
        return self.fit(X, y).transform(X)
