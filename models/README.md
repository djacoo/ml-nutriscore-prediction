# Trained Models Directory

This directory contains all trained machine learning models for the Nutri-Score prediction project.

## Structure

```
models/
├── trained/                    # Trained model artifacts
│   ├── logistic_regression/   # Logistic Regression models
│   ├── knn/                   # K-Nearest Neighbors models
│   ├── svm/                   # Support Vector Machine models
│   ├── random_forest/         # Random Forest models
│   ├── xgboost/              # XGBoost models
│   └── naive_bayes/          # Naive Bayes models
└── preprocessing/             # Preprocessing artifacts (scalers, encoders, etc.)
```

## Naming Convention

Trained models should follow this naming convention:

```
{model_type}_{version}_{optional_description}.joblib
```

Examples:
- `logistic_regression_v1_baseline.joblib`
- `random_forest_v2_tuned.joblib`
- `xgboost_v3_best.joblib`

## Model Files

Each trained model saves two files:

1. **Model file** (`.joblib`): Contains the trained model and all attributes
   - The actual model instance
   - Hyperparameters
   - Training history
   - Metadata

2. **Metadata file** (`_metadata.json`): Human-readable training information
   - Model name and hyperparameters
   - Training/validation/test metrics
   - Training duration
   - Class labels
   - Timestamps

## Usage

### Saving a Model

```python
from models.logistic_regression import LogisticRegressionModel

model = LogisticRegressionModel(C=1.0, max_iter=1000)
model.train(X_train, y_train, X_val, y_val)

# Save to appropriate directory
model.save('models/trained/logistic_regression/model_v1_baseline.joblib')
```

### Loading a Model

```python
from models.logistic_regression import LogisticRegressionModel

model = LogisticRegressionModel()
model.load('models/trained/logistic_regression/model_v1_baseline.joblib')

# Now ready to use
predictions = model.predict(X_test)
metrics = model.evaluate(X_test, y_test)
```

## Model Versions

Keep track of different model versions and their performance:

| Model | Version | Accuracy | F1 (macro) | Notes |
|-------|---------|----------|------------|-------|
| Logistic Regression | v1 | - | - | Baseline model |
| KNN | v1 | - | - | K=5, default parameters |
| SVM | v1 | - | - | RBF kernel |
| Random Forest | v1 | - | - | 100 estimators |
| XGBoost | v1 | - | - | Default parameters |
| Naive Bayes | v1 | - | - | Gaussian NB |

*Update this table as models are trained*

## Best Practices

1. **Version Control**: Always increment version numbers for new training runs
2. **Documentation**: Add descriptive suffixes to model names (e.g., `_tuned`, `_baseline`)
3. **Keep Best Models**: Don't delete previous versions until you're certain new ones are better
4. **Metadata**: Always save metadata alongside models for tracking
5. **Backup**: Consider backing up best-performing models outside the repository

## Git Tracking

**Important**: Model files (`.joblib`) are typically large and should not be committed to git.
The `.gitignore` file is configured to exclude model artifacts while keeping the directory structure.

Only commit:
- Directory structure (.gitkeep files)
- Documentation (README files)
- Small metadata files if needed for reference

For sharing models:
- Use cloud storage (Google Drive, S3, etc.)
- Use model versioning platforms (MLflow, Weights & Biases)
- Use Git LFS for large files if necessary
