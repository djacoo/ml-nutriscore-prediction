"""Download and filter Open Food Facts dataset for Nutri-Score prediction."""

import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

OPENFOODFACTS_URL = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
RAW_FILE = RAW_DATA_DIR / "openfoodfacts_raw.csv.gz"
FILTERED_FILE = PROCESSED_DATA_DIR / "openfoodfacts_filtered.csv"

RELEVANT_COLUMNS = [
    'nutriscore_grade',
    'code',
    'product_name',
    'brands',
    'categories',
    'countries',
    'energy_100g',
    'energy-kcal_100g',
    'fat_100g',
    'saturated-fat_100g',
    'carbohydrates_100g',
    'sugars_100g',
    'fiber_100g',
    'proteins_100g',
    'salt_100g',
    'sodium_100g',
    'fruits-vegetables-nuts_100g',
    'fruits-vegetables-nuts-estimate_100g',
    'additives_n',
    'ingredients_n',
    'nutrition_grade_fr',
    'pnns_groups_1',
    'pnns_groups_2',
    'main_category',
]


def download_dataset(url, output_path):
    """Download dataset with progress bar."""
    try:
        print(f"Downloading from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)

        print(f"Downloaded to {output_path}")
        return True

    except Exception as e:
        print(f"Download failed: {e}")
        return False


def load_and_filter_data(input_path, output_path, sample_size=None, chunksize=10000):
    """Load raw data, filter for valid Nutri-Score products, and save."""
    try:
        print(f"\nProcessing in chunks of {chunksize:,} rows...")

        total_rows = 0
        filtered_rows = 0
        valid_rows = 0
        grade_counts = {}
        missing_columns = []
        existing_columns = None
        all_filtered_chunks = []

        chunk_iterator = pd.read_csv(
            input_path,
            compression='gzip',
            sep='\t',
            low_memory=False,
            on_bad_lines='skip',
            chunksize=chunksize
        )

        for df_chunk in tqdm(chunk_iterator, desc="Processing chunks"):
            total_rows += len(df_chunk)

            if existing_columns is None:
                existing_columns = [col for col in RELEVANT_COLUMNS if col in df_chunk.columns]
                missing_columns = [col for col in RELEVANT_COLUMNS if col not in df_chunk.columns]

            df_filtered = df_chunk[df_chunk['nutriscore_grade'].notna()].copy()
            filtered_rows += len(df_filtered)

            if len(df_filtered) == 0:
                continue

            df_filtered = df_filtered[existing_columns]

            valid_grades = ['a', 'b', 'c', 'd', 'e']
            df_filtered = df_filtered[df_filtered['nutriscore_grade'].str.lower().isin(valid_grades)]
            valid_rows += len(df_filtered)

            if len(df_filtered) == 0:
                continue

            chunk_grade_counts = df_filtered['nutriscore_grade'].str.upper().value_counts()
            for grade, count in chunk_grade_counts.items():
                grade_counts[grade] = grade_counts.get(grade, 0) + count

            all_filtered_chunks.append(df_filtered)

        print(f"Processed {total_rows:,} total rows")
        print(f"Found {filtered_rows:,} with Nutri-Score labels")
        print(f"Kept {valid_rows:,} with valid grades (a-e)")

        if missing_columns:
            print(f"\nMissing columns:")
            for col in missing_columns:
                print(f"  - {col}")

        print(f"Kept {len(existing_columns)} columns")

        df_filtered = pd.concat(all_filtered_chunks, ignore_index=True)

        if sample_size and len(df_filtered) > sample_size:
            df_filtered = df_filtered.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size:,} products")
            grade_counts = df_filtered['nutriscore_grade'].str.upper().value_counts().sort_index()

        print(f"\nSaving to {output_path}...")
        df_filtered.to_csv(output_path, index=False)

        print("\nNutri-Score distribution:")
        grade_counts_sorted = dict(sorted(grade_counts.items()))
        for grade, count in grade_counts_sorted.items():
            percentage = (count / valid_rows) * 100 if valid_rows > 0 else 0
            print(f"  Grade {grade}: {count:,} ({percentage:.1f}%)")

        metadata = {
            'total_products': total_rows,
            'filtered_products': valid_rows,
            'columns': len(existing_columns),
            'missing_columns': missing_columns,
            'grade_distribution': grade_counts_sorted,
            'data_source': OPENFOODFACTS_URL,
            'download_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        metadata_path = PROCESSED_DATA_DIR / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_path}")

        return df_filtered

    except Exception as e:
        print(f"Processing failed: {e}")
        raise


def main():
    print("=" * 70)
    print("Open Food Facts Data Download")
    print("=" * 70)

    if FILTERED_FILE.exists():
        print(f"\nFiltered dataset already exists at {FILTERED_FILE}")
        print("Delete it first if you want to re-download.")
        return

    if not RAW_FILE.exists():
        success = download_dataset(OPENFOODFACTS_URL, RAW_FILE)
        if not success:
            print("\nDownload failed. Exiting.")
            return
    else:
        print(f"Raw file already exists at {RAW_FILE}")

    df = load_and_filter_data(RAW_FILE, FILTERED_FILE, sample_size=100000)

    print("\n" + "=" * 70)
    print("Download Complete")
    print("=" * 70)
    print(f"\nFiltered dataset: {FILTERED_FILE}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()
