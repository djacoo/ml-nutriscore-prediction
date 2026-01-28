"""Feature engineering for nutritional data."""
import joblib
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates derived features from nutritional values."""

    def __init__(
        self,
        add_ratios: bool = True,
        add_energy_density: bool = True,
        add_caloric_contributions: bool = True,
        add_boolean_flags: bool = True
    ):
        self.add_ratios = add_ratios
        self.add_energy_density = add_energy_density
        self.add_caloric_contributions = add_caloric_contributions
        self.add_boolean_flags = add_boolean_flags

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEngineer':
        """No fitting needed, just returns self."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to the dataframe."""
        X_engineered = X.copy()

        if self.add_ratios:
            X_engineered = self._add_ratios(X_engineered)

        if self.add_energy_density:
            X_engineered = self._add_energy_density(X_engineered)

        if self.add_caloric_contributions:
            X_engineered = self._add_caloric_contributions(X_engineered)

        if self.add_boolean_flags:
            X_engineered = self._add_boolean_flags(X_engineered)

        # Fill NaN from division by zero
        derived_cols = [col for col in X_engineered.columns if col not in X.columns]
        if derived_cols:
            for col in derived_cols:
                if X_engineered[col].isna().any():
                    X_engineered[col] = X_engineered[col].fillna(0.0)

        return X_engineered

    def _add_ratios(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate ratios between macro nutrients."""
        if 'fat_100g' in X.columns and 'proteins_100g' in X.columns:
            X['fat_to_protein_ratio'] = np.divide(
                X['fat_100g'], X['proteins_100g'],
                out=np.full_like(X['fat_100g'], np.nan, dtype=float),
                where=X['proteins_100g'] != 0
            )

        if 'sugars_100g' in X.columns and 'carbohydrates_100g' in X.columns:
            X['sugar_to_carb_ratio'] = np.divide(
                X['sugars_100g'], X['carbohydrates_100g'],
                out=np.full_like(X['sugars_100g'], np.nan, dtype=float),
                where=X['carbohydrates_100g'] != 0
            )

        if 'saturated-fat_100g' in X.columns and 'fat_100g' in X.columns:
            X['saturated_to_total_fat_ratio'] = np.divide(
                X['saturated-fat_100g'], X['fat_100g'],
                out=np.full_like(X['saturated-fat_100g'], np.nan, dtype=float),
                where=X['fat_100g'] != 0
            )

        return X

    def _add_energy_density(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate energy per gram."""
        if 'energy-kcal_100g' in X.columns:
            X['energy_density'] = X['energy-kcal_100g'] / 100.0
        return X

    def _add_caloric_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate calories from each macro (fat=9kcal/g, carbs/protein=4kcal/g)."""
        if 'fat_100g' in X.columns:
            X['calories_from_fat'] = X['fat_100g'] * 9.0

        if 'carbohydrates_100g' in X.columns:
            X['calories_from_carbs'] = X['carbohydrates_100g'] * 4.0

        if 'proteins_100g' in X.columns:
            X['calories_from_protein'] = X['proteins_100g'] * 4.0

        return X

    def _add_boolean_flags(self, X: pd.DataFrame) -> pd.DataFrame:
        """Flag products with high fat/sugar/salt based on WHO thresholds."""
        if 'fat_100g' in X.columns:
            X['high_fat'] = (X['fat_100g'] > 20.0).astype(int)

        if 'sugars_100g' in X.columns:
            X['high_sugar'] = (X['sugars_100g'] > 15.0).astype(int)

        if 'salt_100g' in X.columns:
            X['high_salt'] = (X['salt_100g'] > 1.5).astype(int)

        return X

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit and transform in one go."""
        return self.fit(X, y).transform(X)

    def save(self, path: str) -> None:
        """Save to file."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'FeatureEngineer':
        """Load from file."""
        return joblib.load(path)
