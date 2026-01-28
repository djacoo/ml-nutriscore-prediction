"""Data preprocessing for missing values and country name cleaning."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import json
import sys
import pycountry

try:
    from .countries_mappings import COUNTRY_OVERRIDES
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.countries_mappings import COUNTRY_OVERRIDES


class MissingValueHandler:
    """Handles missing values through imputation and feature removal."""

    def __init__(self, threshold_drop_feature: float = 0.95):
        self.threshold_drop_feature = threshold_drop_feature
        self.imputation_stats = {}
        self.dropped_features = []
        self.missing_report = {}

    def analyze_missing_values(self, df: pd.DataFrame) -> Dict:
        """Analyze missing value patterns."""
        missing_info = {}

        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100

            missing_info[col] = {
                'missing_count': int(missing_count),
                'missing_percentage': round(missing_pct, 2),
                'dtype': str(df[col].dtype),
                'unique_values': int(df[col].nunique()) if missing_count < len(df) else 0
            }

        missing_info = dict(sorted(missing_info.items(),
                                  key=lambda x: x[1]['missing_percentage'],
                                  reverse=True))

        return missing_info

    def handle_missing_values(self, df: pd.DataFrame,
                            target_col: str = 'nutriscore_grade') -> pd.DataFrame:
        """Handle missing values based on strategy."""
        df_processed = df.copy()
        initial_rows, initial_cols = len(df_processed), len(df_processed.columns)

        print(f"\nMissing Value Handling: {initial_rows:,} rows × {initial_cols} columns")

        self.missing_report = self.analyze_missing_values(df_processed)

        top_missing = [(col, info) for col, info in list(self.missing_report.items())[:5]
                       if info['missing_percentage'] > 0]
        if top_missing:
            print("Top missing features:")
            for col, info in top_missing:
                print(f"  {col}: {info['missing_percentage']:.1f}% ({info['missing_count']:,})")

        threshold_pct = self.threshold_drop_feature * 100
        high_missing_cols = [col for col, info in self.missing_report.items()
                            if info['missing_percentage'] > threshold_pct and col != target_col]

        if high_missing_cols:
            df_processed = df_processed.drop(columns=high_missing_cols)
            self.dropped_features.extend(high_missing_cols)
            print(f"Dropped {len(high_missing_cols)} features with >{threshold_pct:.0f}% missing")

        target_missing = df_processed[target_col].isnull().sum()
        if target_missing > 0:
            df_processed = df_processed[df_processed[target_col].notna()]
            print(f"Dropped {target_missing:,} rows with missing target")

        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        imputed_numerical = []

        for col in numerical_cols:
            missing_count = df_processed[col].isnull().sum()
            if missing_count > 0:
                if col == 'additives_n':
                    df_processed[col] = df_processed[col].fillna(0)
                    method, value = 'constant', 0
                else:
                    value = df_processed[col].median()
                    df_processed[col] = df_processed[col].fillna(value)
                    method = 'median'

                self.imputation_stats[col] = {
                    'method': method,
                    'value': float(value) if method == 'median' else value,
                    'missing_count': int(missing_count),
                    'rationale': 'Assume no additives' if col == 'additives_n' else 'Median imputation'
                }
                imputed_numerical.append((col, method, value, missing_count))

        if imputed_numerical:
            print(f"Imputed {len(imputed_numerical)} numerical features:")
            for col, method, value, count in imputed_numerical[:5]:
                val_str = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
                print(f"  {col}: {method} ({val_str}) - {count:,} values")

        categorical_cols = [col for col in df_processed.select_dtypes(include=['object']).columns
                           if col != target_col]
        imputed_categorical = []

        for col in categorical_cols:
            missing_count = df_processed[col].isnull().sum()
            if missing_count > 0:
                df_processed[col] = df_processed[col].fillna('unknown')
                self.imputation_stats[col] = {
                    'method': 'constant',
                    'value': 'unknown',
                    'missing_count': int(missing_count),
                    'rationale': 'Placeholder for missing categorical data'
                }
                imputed_categorical.append((col, missing_count))

        if imputed_categorical:
            print(f"Imputed {len(imputed_categorical)} categorical features with 'unknown'")

        final_rows, final_cols = len(df_processed), len(df_processed.columns)
        remaining_missing = df_processed.isnull().sum().sum()

        print(f"\nResult: {final_rows:,} rows × {final_cols} columns")
        if initial_rows != final_rows:
            print(f"  Rows removed: {initial_rows - final_rows:,} ({(initial_rows - final_rows)/initial_rows*100:.1f}%)")
        if initial_cols != final_cols:
            print(f"  Columns removed: {initial_cols - final_cols}")

        if remaining_missing > 0:
            print(f"  Warning: {remaining_missing:,} missing values remaining")
        else:
            print("  All missing values handled")

        return df_processed

    def save_imputation_report(self, output_path: Path) -> None:
        """Save report to JSON."""
        report = {
            'strategy': {
                'threshold_drop_feature': self.threshold_drop_feature,
                'description': 'Drop features >95% missing, impute others based on type'
            },
            'dropped_features': self.dropped_features,
            'imputation_statistics': self.imputation_stats,
            'missing_value_analysis': self.missing_report
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Saved imputation report: {output_path}")


def normalize_single_country(raw_name: str) -> str:
    """Convert raw country name to standard ISO name."""
    name = str(raw_name).lower().strip()
    name = name.replace("en:", "").replace("-", " ")

    if not name:
        return None

    if name in COUNTRY_OVERRIDES:
        return COUNTRY_OVERRIDES[name]

    try:
        country = pycountry.countries.lookup(name)
        country_name = country.name

        country_name_lower = country_name.lower()

        if country_name_lower in COUNTRY_OVERRIDES:
            return COUNTRY_OVERRIDES[country_name_lower]

        standard_names = set(COUNTRY_OVERRIDES.values())
        for std_name in standard_names:
            if country_name_lower == std_name.lower():
                return std_name

        return country_name
    except LookupError:
        return None


def clean_countries_column(entry) -> str:
    """Clean and normalize country entries."""
    if pd.isna(entry) or entry == "unknown":
        return "unknown"

    valid_countries = set()

    for raw_item in str(entry).split(','):
        clean_name = normalize_single_country(raw_item)
        if clean_name:
            valid_countries.add(clean_name)

    if not valid_countries:
        return "unknown"

    return ",".join(sorted(valid_countries))


def preprocess_dataset(input_path: Path,
                      output_path: Path,
                      report_path: Path = None) -> Tuple[pd.DataFrame, MissingValueHandler]:
    """Main preprocessing function."""
    print(f"\nLoading data: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded: {len(df):,} rows × {len(df.columns)} columns")

    handler = MissingValueHandler(threshold_drop_feature=0.95)
    df_processed = handler.handle_missing_values(df)

    if 'countries' in df_processed.columns:
        print("\nCleaning countries column...")
        df_processed['countries'] = df_processed['countries'].apply(clean_countries_column)
        print("Countries column cleaned")

    columns_to_remove = []

    if 'energy_100g' in df_processed.columns:
        columns_to_remove.append('energy_100g')

    for col in ['main_category', 'categories']:
        if col in df_processed.columns and df_processed[col].nunique() > 10000:
            columns_to_remove.append(col)

    if columns_to_remove:
        df_processed = df_processed.drop(columns=columns_to_remove)
        print(f"Removed {len(columns_to_remove)} unnecessary column(s)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    print(f"\nSaved processed data: {output_path}")

    if report_path:
        handler.save_imputation_report(report_path)

    return df_processed, handler


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    input_file = project_root / "data" / "processed" / "openfoodfacts_filtered.csv"
    output_file = project_root / "data" / "processed" / "openfoodfacts_preprocessed.csv"
    report_file = project_root / "data" / "processed" / "imputation_report.json"

    df_processed, handler = preprocess_dataset(input_file, output_file, report_file)
    print(f"\nPreprocessing complete: {df_processed.shape}")
