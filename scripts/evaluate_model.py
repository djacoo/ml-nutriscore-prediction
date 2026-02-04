import argparse
import sys
from pathlib import Path
import pandas as pd
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.models import ModelRegistry

"""
This script evaluates a trained model on a test set.
It loads the test set and the trained model, and evaluates the model on the test set.
It displays the evaluation metrics and the classification report.
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
    parser = argparse.ArgumentParser(description='Evaluate a trained model')
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to the trained model file (.joblib)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        required=True,
        help='Type of model (e.g., logistic_regression, knn, svm, random_forest, xgboost, naive_bayes)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/splits',
        help='Directory containing test data (default: data/splits)'
    )
    parser.add_argument(
        '--show-report',
        action='store_true',
        help='Show detailed classification report'
    )
    parser.add_argument(
        '--show-confusion-matrix',
        action='store_true',
        help='Show confusion matrix'
    )

    args = parser.parse_args()

    print_header("Model Evaluation")

    model_path = Path(args.model_path)
    if not model_path.exists():
        print_error(f"Model file '{model_path}' does not exist.")
        sys.exit(1)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print_error(f"Data directory '{data_dir}' does not exist.")
        sys.exit(1)

    print_step("1/4", "Loading test data")
    print_info("Loading test set to measure model generalization")
    print()
    try:
        X_test = pd.read_csv(data_dir / 'X_test.csv')
        y_test = pd.read_csv(data_dir / 'y_test.csv').values.ravel()

        non_numeric_cols = X_test.select_dtypes(include=['object']).columns.tolist()
        if non_numeric_cols:
            print_info(f"Dropping non-numeric columns: {non_numeric_cols}")
            X_test = X_test.select_dtypes(include=['number'])

        print_success(f"Test set: {X_test.shape[0]:,} samples, {X_test.shape[1]} features")
    except FileNotFoundError as e:
        print_error(f"Could not find test data: {e}")
        sys.exit(1)
    print()

    print_step("2/4", "Loading trained model")
    try:
        model = ModelRegistry.create_model(args.model_type)
        model.load(model_path)
        print_success(f"Model loaded: {model_path.name}")
    except Exception as e:
        print_error(f"Error loading model: {e}")
        sys.exit(1)
    print()

    metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
    if metadata_path.exists():
        print_step("3/4", "Model information")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print_success(f"Model Name: {metadata.get('model_name', 'N/A')}")
        print_success(f"Hyperparameters: {metadata.get('hyperparameters', {})}")

        history = metadata.get('training_history', {})
        if history.get('training_time'):
            print_success(f"Training Time: {history['training_time']:.2f}s")

        train_metrics = history.get('train_metrics', {})
        val_metrics = history.get('val_metrics', {})
        if train_metrics and val_metrics:
            train_acc = train_metrics.get('accuracy', 0)
            val_acc = val_metrics.get('accuracy', 0)
            print_success(f"Train Accuracy: {train_acc:.4f}")
            print_success(f"Val Accuracy: {val_acc:.4f}")
            gap = train_acc - val_acc
            if gap > 0.05:
                print_info(f"Train-val gap ({gap:.4f}) suggests possible overfitting")
            else:
                print_info(f"Train-val gap ({gap:.4f}) indicates good generalization")
        print()

    step_num = "4/4" if metadata_path.exists() else "3/4"
    print_step(step_num, "Evaluating on test set")
    print_info("Computing metrics: accuracy (overall), macro (unweighted), weighted (balanced)")
    print()
    test_metrics = model.evaluate(X_test, y_test, dataset_name='test', verbose=True)

    if args.show_report:
        print_step("EXTRA", "Classification Report")
        print_info("Per-class breakdown showing precision, recall, and f1-score for each grade")
        print()
        print(model.get_classification_report(X_test, y_test))
        print()

    if args.show_confusion_matrix:
        print_step("EXTRA", "Confusion Matrix")
        print_info("Rows: true labels | Columns: predicted labels | Diagonal: correct predictions")
        print()
        cm = model.get_confusion_matrix(X_test, y_test)
        print(cm)
        print()

    model_name = model_path.stem
    print(f"{CYAN}{BOX_TL}{BOX_H * 68}{BOX_TR}{NC}")
    print(f"{CYAN}{BOX_V}{NC} {BOLD}{GREEN}Evaluation Complete{NC}{' ' * 48} {CYAN}{BOX_V}{NC}")
    print(f"{CYAN}{BOX_V}{NC}{' ' * 68} {CYAN}{BOX_V}{NC}")
    print(f"{CYAN}{BOX_V}{NC}   Model: {BOLD}{model_name}{NC}{' ' * (60 - len(model_name))} {CYAN}{BOX_V}{NC}")
    print(f"{CYAN}{BOX_V}{NC}{' ' * 68} {CYAN}{BOX_V}{NC}")
    print(f"{CYAN}{BOX_V}{NC}   Test Accuracy: {BOLD}{test_metrics['accuracy']:.4f}{NC}{' ' * 43} {CYAN}{BOX_V}{NC}")
    print(f"{CYAN}{BOX_V}{NC}   Test Precision (weighted): {BOLD}{test_metrics['precision_weighted']:.4f}{NC}{' ' * 28} {CYAN}{BOX_V}{NC}")
    print(f"{CYAN}{BOX_V}{NC}   Test Recall (weighted): {BOLD}{test_metrics['recall_weighted']:.4f}{NC}{' ' * 31} {CYAN}{BOX_V}{NC}")
    print(f"{CYAN}{BOX_V}{NC}   Test F1 (weighted): {BOLD}{test_metrics['f1_weighted']:.4f}{NC}{' ' * 35} {CYAN}{BOX_V}{NC}")
    print(f"{CYAN}{BOX_BL}{BOX_H * 68}{BOX_BR}{NC}")


if __name__ == '__main__':
    main()
