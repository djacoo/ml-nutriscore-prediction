# Preprocessing Pipeline

Complete preprocessing pipeline for Nutri-Score prediction, from raw data to model-ready features.

## Quick Start

To execute the full pipeline from scratch:

```bash
# 1. Download raw data
python scripts/download_data.py

# 2. Run preprocessing (create a script or notebook with the code from "Execution Steps" below)
python scripts/run_preprocessing.py

# 3. Data is now ready in data/splits/ for model training
```

The pipeline takes ~10-15 minutes on the 100k sample. You'll get train/val/test splits with ~70k/15k/15k samples.

## Pipeline Structure

Seven stages transform raw Open Food Facts data:

1. Data Download
2. Missing Value Handling
3. Outlier Removal
4. Categorical Encoding
5. Feature Engineering
6. Scaling
7. Dimensionality Reduction (PCA)

Built with sklearn's `Pipeline` API. Each transformer has `fit()` and `transform()` methods:

```python
Pipeline([
    ('missing_values', MissingValueTransformer),
    ('outlier_removal', OutlierRemovalTransformer),
    ('encoding', FeatureEncoder),
    ('feature_engineering', FeatureEngineer),
    ('scaling', FeatureScaler),
    ('pca', FeatureReducer)
])
```

## Execution Steps

### 1. Download Data

```bash
python scripts/download_data.py
```

Downloads ~7GB compressed dataset from Open Food Facts, filters for valid Nutri-Score labels (a-e), samples 100k products, and saves to `data/processed/openfoodfacts_filtered.csv`.

### 2. Preprocess

```python
import pandas as pd
from pathlib import Path
from features.preprocessing_pipeline import PreprocessingPipeline
from data.data_loader import split_data, save_splits, verify_stratification, save_split_metadata

data_path = Path("data/processed/openfoodfacts_filtered.csv")
df = pd.read_csv(data_path)

pipeline = PreprocessingPipeline(
    missing_threshold=0.95,
    top_n_countries=15,
    scaling_method='auto',
    scaling_skew_threshold=1.0,
    pca_variance_threshold=0.95,
    target_col='nutriscore_grade',
    include_feature_engineering=True,
    remove_statistical_outliers=False,
    include_pca=True
)

df_processed = pipeline.fit_transform(df)

output_path = Path("data/processed/openfoodfacts_preprocessed.csv")
df_processed.to_csv(output_path, index=False)
```

### 3. Split Data

```python
X_train, y_train, X_val, y_val, X_test, y_test = split_data(
    df_processed,
    target_col='nutriscore_grade',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42,
    stratify=True
)

stats = verify_stratification(y_train, y_val, y_test)

output_dir = Path("data/splits")
save_splits(X_train, y_train, X_val, y_val, X_test, y_test, output_dir)

config = {
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'random_state': 42,
    'stratify': True
}
save_split_metadata(stats, config, output_dir / 'split_metadata.json')
```

Data is now ready for training.

## Components

### MissingValueTransformer

[outlier_removal.py:9-40](../src/features/outlier_removal.py#L9-L40)

Two-step strategy:
- Drop columns with >95% missing values
- Impute remaining: numerical → median, `additives_n` → 0, categorical → 'unknown'
- Drop rows where target is missing

### OutlierRemovalTransformer

[outlier_removal.py:43-248](../src/features/outlier_removal.py#L43-L248)

Domain-based validity checks:
- Fat, saturated fat, carbs, sugars, fiber, proteins: 0-100g per 100g
- Salt: 0-50g per 100g

Optional 3×IQR statistical removal (disabled by default).

### FeatureEncoder

[encoding.py:14-163](../src/features/encoding.py#L14-L163)

Three strategies for three features:

- **Countries**: MultiLabelBinarizer for top 15 countries (products can have multiple)
- **pnns_groups_1**: OneHotEncoder for food categories
- **pnns_groups_2**: TargetEncoder for fine-grained categories

### FeatureEngineer

[feature_engineering.py:10-123](../src/features/feature_engineering.py#L10-L123)

Creates 10 derived features:

**Ratios:**
- `sugar_to_carbs_ratio` = sugars / carbohydrates
- `saturated_to_total_fat_ratio` = saturated fat / total fat
- `protein_to_energy_ratio` = proteins / energy (kcal)
- `fiber_to_carbs_ratio` = fiber / carbohydrates

**Energy:**
- `energy_density` = energy (kcal) / 100

**Caloric contributions:**
- `calories_from_fat` = fat × 9
- `calories_from_carbs` = carbohydrates × 4
- `calories_from_protein` = proteins × 4

**WHO thresholds:**
- `high_sugar_flag` = 1 if sugars > 15g/100g
- `high_salt_flag` = 1 if salt > 1.5g/100g

### FeatureScaler

[scaling.py:11-82](../src/features/scaling.py#L11-L82)

Auto-selects per feature:
- Skewed (|skewness| > 1.0): MinMaxScaler
- Symmetric: StandardScaler

Skips metadata columns.

### FeatureReducer

[dimensionality_reduction.py:10-112](../src/features/dimensionality_reduction.py#L10-L112)

PCA that auto-selects components to retain 95% variance. Typically reduces 50-80 features to 20-30 components.

## Data Leakage Prevention

Pipeline fits only on training data:

```python
def fit(self, X, y=None):
    if self.split_group_col in X.columns:
        train_mask = X[self.split_group_col] == 'train'
        X_train = X[train_mask]
        y_train = y[train_mask] if y is not None else X_train[self.target_col]
        self.pipeline.fit(X_train, y_train)
    else:
        self.pipeline.fit(X, y)
```

When a `split_group` column exists, statistics (mean, median, PCA components, target encoding) are computed only on training data. Validation and test sets are transformed using these training statistics.

## Configuration

```python
PreprocessingPipeline(
    missing_threshold=0.95,              # Drop features with more missing
    top_n_countries=15,                  # Countries to keep
    scaling_method='auto',               # 'auto', 'standard', or 'minmax'
    scaling_skew_threshold=1.0,          # Skewness cutoff for auto scaling
    pca_variance_threshold=0.95,         # Variance to retain
    target_col='nutriscore_grade',       # Target column name
    split_group_col='split_group',       # Train/val/test indicator
    preserve_cols=[],                    # Columns to skip
    include_feature_engineering=True,    # Add derived features
    feature_engineering_kwargs=None,     # Custom FeatureEngineer args
    remove_statistical_outliers=False,   # Use statistical outlier removal
    include_pca=True                     # Apply PCA
)
```

## Output Files

After execution:

- `data/raw/openfoodfacts_raw.csv.gz` - Downloaded dataset
- `data/processed/openfoodfacts_filtered.csv` - Filtered products
- `data/processed/metadata.json` - Download stats
- `data/processed/openfoodfacts_preprocessed.csv` - Processed data
- `data/splits/X_train.csv`, `y_train.csv` - Training (70%)
- `data/splits/X_val.csv`, `y_val.csv` - Validation (15%)
- `data/splits/X_test.csv`, `y_test.csv` - Test (15%)
- `data/splits/split_metadata.json` - Split stats

Typical results:
- ~70k train, ~15k val, ~15k test samples
- 20-30 PCA components
- Balanced class distribution
- No missing values, all features scaled
