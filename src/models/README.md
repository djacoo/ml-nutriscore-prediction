# Models Package

This package contains all machine learning model implementations for the Nutri-Score prediction project.

## Structure

```
src/models/
├── __init__.py                 # Package initialization and exports
├── base_model.py              # Abstract base class for all models
├── model_registry.py          # Registry system for model management
├── logistic_regression.py     # Logistic Regression implementation
├── knn.py                     # K-Nearest Neighbors implementation
├── svm.py                     # Support Vector Machine implementation
├── random_forest.py           # Random Forest implementation
├── xgboost_model.py          # XGBoost implementation
└── naive_bayes.py            # Naive Bayes implementation
```

## Base Model Architecture

All models inherit from `BaseModel` which provides:

### Core Methods
- `train()` - Train the model with training and validation data
- `predict()` - Make predictions on new data
- `predict_proba()` - Get probability predictions (if supported)
- `evaluate()` - Evaluate model performance with standard metrics
- `save()` - Save trained model to disk
- `load()` - Load trained model from disk

### Utility Methods
- `get_classification_report()` - Detailed per-class metrics
- `get_confusion_matrix()` - Confusion matrix for predictions
- `get_params()` - Get model hyperparameters
- `set_params()` - Update model hyperparameters
- `get_training_history()` - Access training history and metrics

### Attributes
- `model_name` - Name identifier for the model
- `model` - Underlying sklearn/xgboost model instance
- `is_trained` - Training status flag
- `hyperparameters` - Dictionary of model hyperparameters
- `training_history` - Metrics and metadata from training
- `classes_` - Array of unique class labels

## Model Registry

The `ModelRegistry` class provides centralized model management:

```python
from models import ModelRegistry

# List all available models
ModelRegistry.print_registry()

# Create a model instance
model = ModelRegistry.create_model('logistic_regression', C=1.0, max_iter=1000)

# Get model information
info = ModelRegistry.get_model_info('logistic_regression')
```

### Registering New Models

Use the `@register_model` decorator:

```python
from models import BaseModel, register_model

@register_model(
    name='my_model',
    description='Description of my model',
    category='linear',  # or 'tree-based', 'ensemble', 'distance-based', etc.
    requires_probability=True
)
class MyModel(BaseModel):
    def _build_model(self):
        # Return the underlying model instance
        pass

    def _fit(self, X_train, y_train, X_val, y_val, verbose):
        # Implement training logic
        # Return training metrics dict
        pass
```

## Implementing a New Model

To implement a new model:

1. **Create a new file** in `src/models/` (e.g., `my_model.py`)

2. **Import required classes**:
   ```python
   from models.base_model import BaseModel
   from models.model_registry import register_model
   from sklearn.some_model import SomeModel
   ```

3. **Define your model class**:
   ```python
   @register_model(
       name='my_model',
       description='My custom model',
       category='general'
   )
   class MyModelClass(BaseModel):
       def __init__(self, **hyperparameters):
           super().__init__(model_name='MyModel', **hyperparameters)

       def _build_model(self):
           return SomeModel(**self.hyperparameters)

       def _fit(self, X_train, y_train, X_val, y_val, verbose):
           self.model.fit(X_train, y_train)
           y_pred = self.model.predict(X_train)
           accuracy = accuracy_score(y_train, y_pred)
           return {'accuracy': accuracy}
   ```

4. **Update `__init__.py`** to export your model:
   ```python
   from .my_model import MyModelClass
   __all__.append('MyModelClass')
   ```

## Usage Example

```python
from models.logistic_regression import LogisticRegressionModel
import pandas as pd

# Load data
X_train = pd.read_csv('data/splits/X_train.csv')
y_train = pd.read_csv('data/splits/y_train.csv')
X_val = pd.read_csv('data/splits/X_val.csv')
y_val = pd.read_csv('data/splits/y_val.csv')
X_test = pd.read_csv('data/splits/X_test.csv')
y_test = pd.read_csv('data/splits/y_test.csv')

# Create and train model
model = LogisticRegressionModel(
    C=1.0,
    max_iter=1000,
    multi_class='multinomial',
    solver='lbfgs'
)

# Train with validation set
train_metrics = model.train(
    X_train, y_train,
    X_val, y_val,
    verbose=True
)

# Evaluate on test set
test_metrics = model.evaluate(X_test, y_test, dataset_name='test')

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Get detailed results
report = model.get_classification_report(X_test, y_test)
cm = model.get_confusion_matrix(X_test, y_test)

# Save model
model.save('models/trained/logistic_regression/model_v1.joblib')

# Load model later
loaded_model = LogisticRegressionModel()
loaded_model.load('models/trained/logistic_regression/model_v1.joblib')
```

## Trained Models Storage

Trained models are saved in `models/trained/` organized by model type:

```
models/trained/
├── logistic_regression/
│   ├── model_v1.joblib
│   ├── model_v1_metadata.json
│   └── ...
├── knn/
├── svm/
├── random_forest/
├── xgboost/
└── naive_bayes/
```

Each saved model includes:
- `.joblib` file: The trained model and all its attributes
- `_metadata.json` file: Training history, hyperparameters, and metrics

## Model Categories

Models are organized into categories:

- **linear**: Logistic Regression
- **distance-based**: K-Nearest Neighbors
- **kernel**: Support Vector Machines
- **tree-based**: Decision Trees
- **ensemble**: Random Forest, XGBoost
- **probabilistic**: Naive Bayes

## Metrics

All models track the following metrics during training and evaluation:

- **Accuracy**: Overall classification accuracy
- **Precision**: Macro and weighted averages
- **Recall**: Macro and weighted averages
- **F1-Score**: Macro and weighted averages
- **Confusion Matrix**: For detailed error analysis
- **Classification Report**: Per-class metrics

## Best Practices

1. **Always use validation set**: Pass `X_val` and `y_val` to `train()` method
2. **Save models with versions**: Use descriptive names like `model_v1.joblib`
3. **Document hyperparameters**: Keep notes on what hyperparameters work best
4. **Track experiments**: Use the training history to compare different configurations
5. **Test on test set only once**: Use validation set for tuning, test set for final evaluation
