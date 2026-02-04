import argparse
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.models import ModelRegistry

"""
This script trains a machine learning model for Nutri-Score prediction.
It loads the preprocessed data, initializes the model, trains the model and saves the model.
It also displays the training metrics and the evaluation metrics.
It can work with different models and versions, and with different hyperparameters.
"""

BLUE = '\033[0;34m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
CYAN = '\033[0;36m'
NC = '\033[0m'
BOLD = '\033[1m'

BOX_H = "━"
BOX_V = "┃"
BOX_TL = "┏"
BOX_TR = "┓"
BOX_BL = "┗"
BOX_BR = "┛"


def print_header(text):
    width = 70
    print(f"{CYAN}{BOX_TL}{BOX_H * (width-2)}{BOX_TR}{NC}")
    print(f"{CYAN}{BOX_V}{NC} {BOLD}{text:^{width-4}}{NC} {CYAN}{BOX_V}{NC}")
    print(f"{CYAN}{BOX_BL}{BOX_H * (width-2)}{BOX_BR}{NC}\n")


def print_step(step, desc):
    print(f"{BLUE}━━━ {BOLD}[{step}]{NC} {BLUE}{desc}{NC}")


def print_success(msg):
    print(f"      {GREEN}✓{NC} {msg}")


def print_error(msg):
    print(f"      {YELLOW}✗{NC} {msg}")


def print_info(msg):
    print(f"      {msg}")


def main():
    parser = argparse.ArgumentParser(description='Train a machine learning model')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name (e.g., logistic_regression, knn, svm, random_forest, xgboost, naive_bayes)'
    )
    parser.add_argument(
        '--version',
        type=str,
        default='v1',
        help='Model version (default: v1)'
    )
    parser.add_argument(
        '--description',
        type=str,
        default='',
        help='Model description (default: empty)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/splits',
        help='Directory containing train/val/test splits (default: data/splits)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/trained',
        help='Directory to save trained models (default: models/trained)'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all available models and exit'
    )
    parser.add_argument(
        '--C',
        type=float,
        help='Logistic Regression: Inverse of regularization strength'
    )
    parser.add_argument(
        '--solver',
        type=str,
        help='Logistic Regression: Solver algorithm (lbfgs, saga, etc.)'
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        help='Logistic Regression: Maximum iterations'
    )
    parser.add_argument(
        '--class-weight',
        type=str,
        help='Logistic Regression: Class weight (balanced or None)'
    )

    args = parser.parse_args()

    if args.list_models:
        print()
        ModelRegistry.print_registry()
        return

    start_time = datetime.now()

    print_header(f"Model Training: {args.model.upper()}")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print_error(f"Data directory '{data_dir}' does not exist.")
        print_info("Please run preprocessing first: python scripts/run_preprocessing.py")
        sys.exit(1)

    print_step("1/4", "Loading preprocessed data")
    print_info("Loading train/val splits for supervised learning workflow")
    print()
    try:
        X_train = pd.read_csv(data_dir / 'X_train.csv')
        y_train = pd.read_csv(data_dir / 'y_train.csv').values.ravel()
        X_val = pd.read_csv(data_dir / 'X_val.csv')
        y_val = pd.read_csv(data_dir / 'y_val.csv').values.ravel()

        non_numeric_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        if non_numeric_cols:
            print_info(f"Dropping non-numeric columns: {non_numeric_cols}")
            X_train = X_train.select_dtypes(include=['number'])
            X_val = X_val.select_dtypes(include=['number'])

        print_success(f"Training set: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        print_success(f"Validation set: {X_val.shape[0]:,} samples")
    except FileNotFoundError as e:
        print_error(f"Could not find data files: {e}")
        print_info("Please run preprocessing first: python scripts/run_preprocessing.py")
        sys.exit(1)
    print()

    print_step("2/4", "Initializing model")
    try:
        
        model_kwargs = {}
        if args.C is not None:
            model_kwargs['C'] = args.C
        if args.solver is not None:
            model_kwargs['solver'] = args.solver
        if args.max_iter is not None:
            model_kwargs['max_iter'] = args.max_iter
        if args.class_weight is not None:
            model_kwargs['class_weight'] = None if args.class_weight.lower() == 'none' else args.class_weight

        model = ModelRegistry.create_model(args.model, **model_kwargs)
        print_success(f"Model: {model.model_name}")
        print_success(f"Hyperparameters: {model.get_params()}")
    except KeyError as e:
        print_error(f"{e}")
        print_info("Use --list-models to see available models")
        sys.exit(1)
    print()

    print_step("3/4", "Training model")
    print_info("Fitting model to learn patterns for Nutri-Score prediction (grades a-e)")
    print()
    model.train(X_train, y_train, X_val, y_val, verbose=True)
    print()

    description_suffix = f"_{args.description}" if args.description else ""
    filename = f"{args.model}_{args.version}{description_suffix}.joblib"
    output_dir = Path(args.output_dir) / args.model
    output_path = output_dir / filename

    print_step("4/4", "Saving model")
    model.save(output_path, save_metadata=True)
    print()

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    val_metrics = model.get_training_history().get('val_metrics', {})
    val_acc = val_metrics.get('accuracy', 0)
    val_f1 = val_metrics.get('f1_weighted', 0)

    print(f"{CYAN}{BOX_TL}{BOX_H * 68}{BOX_TR}{NC}")
    print(f"{CYAN}{BOX_V}{NC} {BOLD}{GREEN}Training Complete{NC}{' ' * 51} {CYAN}{BOX_V}{NC}")
    print(f"{CYAN}{BOX_V}{NC}{' ' * 68} {CYAN}{BOX_V}{NC}")
    print(f"{CYAN}{BOX_V}{NC}   Model: {BOLD}{args.model}{NC}{' ' * (60 - len(args.model))} {CYAN}{BOX_V}{NC}")
    print(f"{CYAN}{BOX_V}{NC}   Saved to: {BOLD}{output_path}{NC}{' ' * (54 - len(str(output_path)))} {CYAN}{BOX_V}{NC}")
    print(f"{CYAN}{BOX_V}{NC}{' ' * 68} {CYAN}{BOX_V}{NC}")
    print(f"{CYAN}{BOX_V}{NC}   Val Accuracy: {BOLD}{val_acc:.4f}{NC}{' ' * 45} {CYAN}{BOX_V}{NC}")
    print(f"{CYAN}{BOX_V}{NC}   Val F1 (weighted): {BOLD}{val_f1:.4f}{NC}{' ' * 38} {CYAN}{BOX_V}{NC}")
    print(f"{CYAN}{BOX_V}{NC}{' ' * 68} {CYAN}{BOX_V}{NC}")
    print(f"{CYAN}{BOX_V}{NC}   Time elapsed: {BOLD}{minutes}m {seconds}s{NC}{' ' * (48 - len(f'{minutes}m {seconds}s'))} {CYAN}{BOX_V}{NC}")
    print(f"{CYAN}{BOX_BL}{BOX_H * 68}{BOX_BR}{NC}")
    print()
    print_info("Next: Evaluate on test set with:")
    print_info(f"  python scripts/evaluate_model.py --model-path {output_path} --model-type {args.model}")


if __name__ == '__main__':
    main()
