import pandas as pd
from pathlib import Path
import sys

from features.preprocessing_pipeline import PreprocessingPipeline
from data.data_loader import split_data, save_splits, verify_stratification, save_split_metadata


def main():
    data_path = Path("data/processed/openfoodfacts_filtered.csv")

    if not data_path.exists():
        print(f"      Error: {data_path} not found")
        print("      Run 'python scripts/download_data.py' first")
        sys.exit(1)

    print("      Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"      Loaded {len(df):,} products with {len(df.columns)} features")

    print("      Applying transformations...")
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
    print(f"      Processed: {len(df_processed):,} products, {len(df_processed.columns)} features")

    print("      Saving preprocessed dataset...")
    output_path = Path("data/processed/openfoodfacts_preprocessed.csv")
    df_processed.to_csv(output_path, index=False)

    print("      Creating train/validation/test splits...")
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

    print(f"      Train: {len(X_train):,} | Validation: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"      Preprocessing complete")


if __name__ == "__main__":
    main()
