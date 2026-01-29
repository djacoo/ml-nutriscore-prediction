# Model Training & Evaluation

Quick guide to train and evaluate ML models for Nutri-Score prediction.

---

## Quick Start (Recommended)

### Interactive Workflow

Run the complete training and evaluation pipeline:

```bash
./run-training-and-evaluation.sh
```

**What it does:**
1. Shows menu of available models
2. Trains selected model on training data with validation
3. Evaluates trained model on test set
4. Displays detailed metrics and classification report
5. Saves model to `models/trained/{model_name}/`

**Prerequisites:** Preprocessed data must exist in `data/splits/`. Run `./run-preprocessing.sh` or './run-preprocessing.sh' first if needed.

---

## Manual Training

### Train a Model

```bash
python scripts/train_model.py --model logistic_regression
```

**What happens:**
- Loads train/val data from `data/splits/`
- Trains model on training set with validation monitoring
- Saves model to `models/trained/logistic_regression/logistic_regression_v1.joblib`
- Saves metadata with train/val metrics to `models/trained/logistic_regression/logistic_regression_v1_metadata.json`
- Displays validation accuracy and next steps for testing

### Training Options

```bash
# List available models
python scripts/train_model.py --list-models

# Train with custom version
python scripts/train_model.py --model logistic_regression --version v2

# Train with version and description
python scripts/train_model.py --model logistic_regression --version v2 --description baseline
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model name (required) | - |
| `--version` | Model version | v1 |
| `--description` | Optional description suffix | empty |
| `--data-dir` | Data directory | data/splits |
| `--output-dir` | Output directory | models/trained |

---

## Manual Evaluation

### Evaluate a Saved Model

```bash
python scripts/evaluate_model.py \
    --model-path models/trained/logistic_regression/logistic_regression_v1.joblib \
    --model-type logistic_regression
```

**What happens:**
- Loads test data from `data/splits/`
- Loads saved model
- Evaluates on test set
- Displays performance metrics

### Evaluation Options

```bash
# Show detailed classification report
python scripts/evaluate_model.py \
    --model-path models/trained/logistic_regression/logistic_regression_v1.joblib \
    --model-type logistic_regression \
    --show-report

# Show confusion matrix
python scripts/evaluate_model.py \
    --model-path models/trained/logistic_regression/logistic_regression_v1.joblib \
    --model-type logistic_regression \
    --show-confusion-matrix
```

**Arguments:**

| Argument | Description | Required |
|----------|-------------|----------|
| `--model-path` | Path to .joblib file | Yes |
| `--model-type` | Model type name | Yes |
| `--data-dir` | Test data directory | No (default: data/splits) |
| `--show-report` | Show classification report | No |
| `--show-confusion-matrix` | Show confusion matrix | No |

---

## Understanding Results

### Saved Files

**Model file (.joblib)**: Binary file with trained model, weights, and training history

**Metadata file (_metadata.json)**: JSON file with hyperparameters and metrics

Example metadata:
```json
{
  "model_name": "LogisticRegression",
  "hyperparameters": {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 42
  },
  "training_history": {
    "train_metrics": {"accuracy": 0.5528},
    "val_metrics": {"accuracy": 0.5490},
    "test_metrics": {"accuracy": 0.5485},
    "training_time": 0.41
  },
  "classes": ["a", "b", "c", "d", "e"]
}
```

### Metrics

- **Accuracy**: Percentage of correct predictions
- **Precision**: Correctness of positive predictions
- **Recall**: Coverage of actual positives
- **F1-Score**: Harmonic mean of precision and recall (best for imbalanced data)

All metrics shown as macro (unweighted) and weighted (by class support).

---

## Package Structure

### Core Files

```
src/models/
├── base_model.py           # Abstract base class (train, predict, evaluate, save, load)
├── model_registry.py       # Registry system (@register_model decorator)
├── logistic_regression.py  # Logistic Regression implementation
└── __init__.py

scripts/
├── train_model.py          # Universal training script
└── evaluate_model.py       # Universal evaluation script

run-training-and-evaluation.sh  # Interactive workflow
```

**BaseModel**: Abstract class providing consistent interface (train, predict, evaluate, save, load)

**ModelRegistry**: Central registry for all models. Use `@register_model` decorator to auto-register new models.

**Training Scripts**: Universal scripts that work with any registered model.

---

## Complete Workflow Example

```bash
# 1. Preprocess data (if not already done)
./run-preprocessing.sh

# 2. Train model (trains with validation monitoring)
python scripts/train_model.py --model logistic_regression --version v1

# 3. Evaluate on test set (separate step for unbiased evaluation)
python scripts/evaluate_model.py \
    --model-path models/trained/logistic_regression/logistic_regression_v1.joblib \
    --model-type logistic_regression \
    --show-report \
    --show-confusion-matrix

# 4. View saved metadata
cat models/trained/logistic_regression/logistic_regression_v1_metadata.json
```

**Note:** Training and testing are separate steps following ML best practices - the test set is only used once for final evaluation.

---