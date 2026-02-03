import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.models import RandomForestModel


BLUE = '\033[0;34m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
CYAN = '\033[0;36m'
NC = '\033[0m'
BOLD = '\033[1m'


def print_header(text):
    print(f"\n{CYAN}{'='*70}{NC}")
    print(f"{CYAN}{BOLD}{text:^70}{NC}")
    print(f"{CYAN}{'='*70}{NC}\n")


def print_result(params, cv_mean, cv_std, val_acc, time_taken):
    n_est = params['n_estimators']
    max_d = params['max_depth'] if params['max_depth'] else 'None'
    max_f = params['max_features']
    print(f"{BLUE}n_est={n_est:<4} max_depth={max_d:<5} max_feat={max_f:<5}{NC} | "
          f"CV: {cv_mean:.4f} (±{cv_std:.4f}) | "
          f"Val: {val_acc:.4f} | "
          f"Time: {int(time_taken)}s")


def main():
    parser = argparse.ArgumentParser(description='Tune Random Forest hyperparameters')
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=3,
        help='Number of cross-validation folds (default: 3 for faster tuning)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/splits',
        help='Directory containing train/val splits'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/trained/random_forest',
        help='Directory to save best model'
    )

    args = parser.parse_args()

    print_header("Random Forest Hyperparameter Tuning")

    # Load data
    print(f"{BLUE}Loading data...{NC}")
    data_dir = Path(args.data_dir)
    X_train = pd.read_csv(data_dir / 'X_train.csv').select_dtypes(include=['number'])
    y_train = pd.read_csv(data_dir / 'y_train.csv').values.ravel()
    X_val = pd.read_csv(data_dir / 'X_val.csv').select_dtypes(include=['number'])
    y_val = pd.read_csv(data_dir / 'y_val.csv').values.ravel()

    print(f"{GREEN}✓{NC} Training: {X_train.shape[0]:,} samples")
    print(f"{GREEN}✓{NC} Validation: {X_val.shape[0]:,} samples\n")

    # Define hyperparameter grid - expanded for more detailed search
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [20, 30, 50, None],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    print(f"{YELLOW}Hyperparameter Grid:{NC}")
    print(f"  n_estimators: {param_grid['n_estimators']}")
    print(f"  max_depth: {param_grid['max_depth']}")
    print(f"  max_features: {param_grid['max_features']}")
    print(f"  min_samples_split: {param_grid['min_samples_split']}")
    print(f"  min_samples_leaf: {param_grid['min_samples_leaf']}")
    total_combinations = (len(param_grid['n_estimators']) * len(param_grid['max_depth']) *
                         len(param_grid['max_features']) * len(param_grid['min_samples_split']) *
                         len(param_grid['min_samples_leaf']))
    print(f"  Total combinations: {total_combinations}")
    print(f"  CV folds: {args.cv_folds}")
    print(f"  Estimated time: ~{int(total_combinations * 1.5)} minutes\n")

    # Grid search
    results = []
    best_val_acc = 0
    best_params = None
    best_model = None

    print(f"{BOLD}Starting grid search...{NC}\n")

    for i, (n_est, max_d, max_f, min_split, min_leaf) in enumerate(product(
        param_grid['n_estimators'],
        param_grid['max_depth'],
        param_grid['max_features'],
        param_grid['min_samples_split'],
        param_grid['min_samples_leaf']
    ), 1):
        params = {
            'n_estimators': n_est,
            'max_depth': max_d,
            'max_features': max_f,
            'min_samples_split': min_split,
            'min_samples_leaf': min_leaf
        }

        print(f"{CYAN}[{i}/{total_combinations}]{NC} Testing: n_est={n_est}, max_depth={max_d}, max_feat={max_f}, min_split={min_split}, min_leaf={min_leaf}")

        start_time = datetime.now()

        # Train model with CV
        model = RandomForestModel(
            n_estimators=n_est,
            max_depth=max_d,
            max_features=max_f,
            min_samples_split=min_split,
            min_samples_leaf=min_leaf,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )

        model.train(X_train, y_train, X_val, y_val, cv_folds=args.cv_folds, verbose=False)

        # Get metrics
        cv_scores = model.get_training_history()['cv_scores']
        cv_mean = cv_scores['mean']
        cv_std = cv_scores['std']
        val_acc = model.get_training_history()['val_metrics']['accuracy']

        elapsed = (datetime.now() - start_time).total_seconds()

        # Store results
        results.append({
            'n_estimators': n_est,
            'max_depth': max_d,
            'max_features': max_f,
            'min_samples_split': min_split,
            'min_samples_leaf': min_leaf,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'val_acc': val_acc,
            'time': elapsed
        })

        print_result(params, cv_mean, cv_std, val_acc, elapsed)

        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params
            best_model = model
            print(f"  {GREEN}★ New best!{NC}")

        print()

    # Summary
    print_header("Tuning Results Summary")

    print(f"{BOLD}All Results (sorted by validation accuracy):{NC}\n")
    results_sorted = sorted(results, key=lambda x: x['val_acc'], reverse=True)

    for i, res in enumerate(results_sorted, 1):
        marker = "★" if i == 1 else " "
        max_d_str = str(res['max_depth']) if res['max_depth'] else 'None'
        print(f"{marker} {i}. n_est={res['n_estimators']:<4} max_d={max_d_str:<5} max_f={res['max_features']:<5} "
              f"min_split={res['min_samples_split']} min_leaf={res['min_samples_leaf']} | "
              f"CV: {res['cv_mean']:.4f} | Val: {res['val_acc']:.4f}")

    print(f"\n{GREEN}{BOLD}Best Parameters:{NC}")
    print(f"  n_estimators: {best_params['n_estimators']}")
    print(f"  max_depth: {best_params['max_depth']}")
    print(f"  max_features: {best_params['max_features']}")
    print(f"  min_samples_split: {best_params['min_samples_split']}")
    print(f"  min_samples_leaf: {best_params['min_samples_leaf']}")
    print(f"  Validation Accuracy: {best_val_acc:.4f}")

    # Save best model
    output_dir = Path(args.output_dir)
    output_path = output_dir / "random_forest_v2_tuned.joblib"
    best_model.save(output_path, save_metadata=True)

    print(f"\n{GREEN}✓{NC} Best model saved to: {output_path}")

    # Save tuning results
    results_path = output_dir / "tuning_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_val_acc': best_val_acc,
            'all_results': results_sorted
        }, f, indent=2)

    print(f"{GREEN}✓{NC} Tuning results saved to: {results_path}")


if __name__ == '__main__':
    main()
