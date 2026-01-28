"""Outlier detection and removal transformers."""
import pandas as pd
from typing import Optional, Dict
from sklearn.base import BaseEstimator, TransformerMixin

from data.preprocessing import MissingValueHandler


class MissingValueTransformer(BaseEstimator, TransformerMixin):
    """Sklearn wrapper for MissingValueHandler."""

    def __init__(
        self, threshold_drop_feature: float = 0.95, target_col: str = 'nutriscore_grade'
    ):
        self.threshold_drop_feature = threshold_drop_feature
        self.target_col = target_col
        self.handler = MissingValueHandler(threshold_drop_feature=threshold_drop_feature)
        self.dropped_features_: Optional[list] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'MissingValueTransformer':
        """Analyze missing patterns."""
        self.handler.analyze_missing_values(X)
        self.dropped_features_ = self.handler.dropped_features.copy()
        return self

    def transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Handle missing values."""
        X_work = X.copy()
        if self.target_col not in X_work.columns and y is not None:
            X_work[self.target_col] = y.values

        return self.handler.handle_missing_values(X_work, target_col=self.target_col)

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit and transform."""
        return self.fit(X, y).transform(X, y)


class OutlierRemovalTransformer(BaseEstimator, TransformerMixin):
    """Removes outliers using domain rules and statistical methods."""

    def __init__(
        self,
        target_col: str = 'nutriscore_grade',
        remove_statistical_outliers: bool = False
    ):
        self.target_col = target_col
        self.remove_statistical_outliers = remove_statistical_outliers

        self.valid_ranges_ = {
            'fat_100g': (0, 100),
            'saturated-fat_100g': (0, 100),
            'carbohydrates_100g': (0, 100),
            'sugars_100g': (0, 100),
            'fiber_100g': (0, 100),
            'proteins_100g': (0, 100),
            'salt_100g': (0, 50),
        }

        self.rows_removed_ = 0
        self.outlier_report_: Optional[Dict] = None

    def _detect_outliers(self, df: pd.DataFrame) -> Dict:
        """Detect outliers in dataset."""
        outlier_info = {}

        for col in self.valid_ranges_.keys():
            if col not in df.columns:
                continue

            col_info = {}

            negative_mask = df[col] < 0
            col_info['negative_count'] = int(negative_mask.sum())

            min_val, max_val = self.valid_ranges_[col]
            below_min = (df[col] < min_val).sum()
            above_max = (df[col] > max_val).sum()
            col_info['below_min'] = int(below_min)
            col_info['above_max'] = int(above_max)
            col_info['valid_range'] = self.valid_ranges_[col]

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            statistical_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            col_info['statistical_outliers'] = int(statistical_outliers)
            col_info['IQR_bounds'] = (float(lower_bound), float(upper_bound))

            col_info['min'] = float(df[col].min())
            col_info['max'] = float(df[col].max())
            col_info['median'] = float(df[col].median())

            outlier_info[col] = col_info

        return outlier_info

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'OutlierRemovalTransformer':
        """Analyze outlier patterns."""
        self.outlier_report_ = {}
        self.outlier_report_['before_cleaning'] = self._detect_outliers(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from data."""
        df_clean = X.copy()
        initial_rows = len(df_clean)

        print("\n" + "="*70)
        print("OUTLIER REMOVAL")
        print("="*70)
        print(f"\nInitial dataset: {initial_rows:,} rows")

        if self.outlier_report_ is None:
            self.outlier_report_ = {}
        if 'before_cleaning' not in self.outlier_report_:
            print("\n1. Analyzing outliers...")
            self.outlier_report_['before_cleaning'] = self._detect_outliers(df_clean)
        else:
            print("\n1. Using pre-analyzed outlier patterns...")

        print("\nOutlier Summary (Before Cleaning):")
        print("-" * 70)
        total_issues = 0
        for col, info in self.outlier_report_['before_cleaning'].items():
            issues = info['negative_count'] + info['above_max']
            if issues > 0:
                total_issues += issues
                print(f"{col:25s}: {info['negative_count']:6,} negative, "
                      f"{info['above_max']:6,} above max ({info['valid_range'][1]})")

        print(f"\nTotal problematic values: {total_issues:,}")

        print("\n2. Removing invalid values...")
        rows_removed = 0
        removal_reasons = {}

        for col in self.valid_ranges_.keys():
            if col not in df_clean.columns:
                continue

            min_val, max_val = self.valid_ranges_[col]

            invalid_mask = (df_clean[col] < min_val) | (df_clean[col] > max_val)
            invalid_count = invalid_mask.sum()

            if invalid_count > 0:
                removal_reasons[col] = {
                    'count': int(invalid_count),
                    'reason': f'Outside valid range ({min_val}, {max_val})'
                }

                df_clean = df_clean[~invalid_mask]
                rows_removed += invalid_count

                print(f"   - {col:25s}: Removed {invalid_count:,} rows")

        if self.remove_statistical_outliers:
            print("\n3. Removing extreme statistical outliers...")

            nutritional_cols = list(self.valid_ranges_.keys())
            existing_cols = [col for col in nutritional_cols if col in df_clean.columns]

            combined_extreme_mask = pd.Series(False, index=df_clean.index)

            for col in existing_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                statistical_lower = Q1 - 3 * IQR
                statistical_upper = Q3 + 3 * IQR

                min_val, max_val = self.valid_ranges_[col]

                lower_bound = max(0, statistical_lower)
                upper_bound = min(statistical_upper, max_val)

                extreme_mask = (df_clean[col] > upper_bound)
                extreme_count = extreme_mask.sum()

                if extreme_count > 0:
                    combined_extreme_mask = combined_extreme_mask | extreme_mask

                    if col not in removal_reasons:
                        removal_reasons[col] = {
                            'count': int(extreme_count),
                            'reason': f'Extreme statistical outlier (3*IQR, upper bound: {upper_bound:.2f})'
                        }

                    print(f"   - {col:25s}: {extreme_count:,} extreme outliers detected "
                          f"(removing values > {upper_bound:.2f}, "
                          f"IQR: {statistical_lower:.2f} - {statistical_upper:.2f}, "
                          f"clamped: {lower_bound:.2f} - {upper_bound:.2f})")

            if combined_extreme_mask.any():
                statistical_outliers_count = combined_extreme_mask.sum()
                df_clean = df_clean[~combined_extreme_mask]
                rows_removed += statistical_outliers_count
                print(f"\n   Removed {statistical_outliers_count:,} rows with statistical outliers")
            else:
                print("\n   No statistical outliers to remove")
        else:
            print("\n3. Statistical outlier removal skipped (remove_statistical_outliers=False)")

        print("\n4. Final verification...")
        final_rows = len(df_clean)
        total_removed = initial_rows - final_rows
        removal_pct = (total_removed / initial_rows) * 100

        print(f"   Initial rows: {initial_rows:,}")
        print(f"   Final rows: {final_rows:,}")
        print(f"   Rows removed: {total_removed:,} ({removal_pct:.2f}%)")

        self.outlier_report_['after_cleaning'] = self._detect_outliers(df_clean)
        self.outlier_report_['removal_summary'] = removal_reasons
        self.outlier_report_['rows_removed'] = int(total_removed)
        self.outlier_report_['removal_percentage'] = float(removal_pct)

        print("\n5. Validation check...")
        issues_remaining = 0
        for col, info in self.outlier_report_['after_cleaning'].items():
            issues = info['negative_count'] + info['above_max']
            if issues > 0:
                issues_remaining += issues
                print(f"   Warning: {col}: {issues:,} issues remaining")

        if issues_remaining == 0:
            print("   All invalid values successfully removed!")
        else:
            print(f"   WARNING: {issues_remaining:,} issues remaining")

        print("\n" + "="*70)

        self.rows_removed_ = total_removed
        return df_clean

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit and transform."""
        return self.fit(X, y).transform(X)
