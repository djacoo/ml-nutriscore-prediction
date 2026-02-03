import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.models import SVMModel


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
    print(f"{BLUE}C={params['C']:<6} gamma={str(params['gamma']):<8}{NC} | "
          f"CV: {cv_mean:.4f} (±{cv_std:.4f}) | "
          f"Val: {val_acc:.4f} | "
          f"Time: {int(time_taken)}s")


def main():
    parser = argparse.ArgumentParser(description='Tune SVM hyperparameters')
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
        default='models/trained/svm',
        help='Directory to save best model'
    )

    args = parser.parse_args()

    print_header("SVM Hyperparameter Tuning")

    # Load data
    print(f"{BLUE}Loading data...{NC}")
    data_dir = Path(args.data_dir)
    X_train = pd.read_csv(data_dir / 'X_train.csv').select_dtypes(include=['number'])
    y_train = pd.read_csv(data_dir / 'y_train.csv').values.ravel()
    X_val = pd.read_csv(data_dir / 'X_val.csv').select_dtypes(include=['number'])
    y_val = pd.read_csv(data_dir / 'y_val.csv').values.ravel()

    print(f"{GREEN}✓{NC} Training: {X_train.shape[0]:,} samples")
    print(f"{GREEN}✓{NC} Validation: {X_val.shape[0]:,} samples\n")

    # Define hyperparameter grid
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'gamma': ['scale', 0.01, 0.1]
    }

    print(f"{YELLOW}Hyperparameter Grid:{NC}")
    print(f"  C: {param_grid['C']}")
    print(f"  gamma: {param_grid['gamma']}")
    print(f"  Total combinations: {len(param_grid['C']) * len(param_grid['gamma'])}")
    print(f"  CV folds: {args.cv_folds}")
    print(f"  Estimated time: ~{len(param_grid['C']) * len(param_grid['gamma']) * 8} minutes\n")

    # Grid search
    results = []
    best_val_acc = 0
    best_params = None
    best_model = None

    print(f"{BOLD}Starting grid search...{NC}\n")

    for i, (C, gamma) in enumerate(product(param_grid['C'], param_grid['gamma']), 1):
        params = {'C': C, 'gamma': gamma}

        print(f"{CYAN}[{i}/{len(param_grid['C']) * len(param_grid['gamma'])}]{NC} Testing: C={C}, gamma={gamma}")

        start_time = datetime.now()

        # Train model with CV
        model = SVMModel(
            C=C,
            gamma=gamma,
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            cache_size=2000,
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
            'C': C,
            'gamma': gamma,
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
        print(f"{marker} {i}. C={res['C']:<6} gamma={str(res['gamma']):<8} | "
              f"CV: {res['cv_mean']:.4f} | Val: {res['val_acc']:.4f}")

    print(f"\n{GREEN}{BOLD}Best Parameters:{NC}")
    print(f"  C: {best_params['C']}")
    print(f"  gamma: {best_params['gamma']}")
    print(f"  Validation Accuracy: {best_val_acc:.4f}")

    # Save best model
    output_dir = Path(args.output_dir)
    output_path = output_dir / "svm_v2_tuned.joblib"
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
