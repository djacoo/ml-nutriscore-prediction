"""
Models package for nutriscore prediction.

This package contains all machine learning model implementations,
including the base model class and specific model implementations.

Available Models:
    - LogisticRegressionModel: Baseline logistic regression model
    - KNNModel: K-Nearest Neighbors classifier
    - SVMModel: Support Vector Machine classifier
    - RandomForestModel: Random Forest ensemble classifier
    - XGBoostModel: XGBoost gradient boosting classifier
    - NaiveBayesModel: Naive Bayes classifier

Usage:
    from models import BaseModel, ModelRegistry
    from models.logistic_regression import LogisticRegressionModel

    # Create a model instance
    model = LogisticRegressionModel(C=1.0, max_iter=1000)

    # Or use the registry
    model = ModelRegistry.create_model('logistic_regression', C=1.0, max_iter=1000)

    # Train the model
    model.train(X_train, y_train, X_val, y_val)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate
    metrics = model.evaluate(X_test, y_test)

    # Save the model
    model.save('models/trained/logistic_regression/model.joblib')
"""

from .base_model import BaseModel
from .model_registry import ModelRegistry, register_model

__all__ = [
    'BaseModel',
    'ModelRegistry',
    'register_model',
]

# Model classes will be added to __all__ as they are implemented:
# 'LogisticRegressionModel',
# 'KNNModel',
# 'SVMModel',
# 'RandomForestModel',
# 'XGBoostModel',
# 'NaiveBayesModel',
