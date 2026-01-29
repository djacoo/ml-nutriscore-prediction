from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


class BaseModel(ABC):

    def __init__(self, model_name: str, **hyperparameters):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.hyperparameters = hyperparameters
        self.training_history = {
            'train_metrics': {},
            'val_metrics': {},
            'test_metrics': {},
            'training_time': None,
            'created_at': datetime.now().isoformat(),
            'last_trained': None
        }
        self.classes_ = None

    @abstractmethod
    def _build_model(self) -> Any:
        pass

    def train(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        if verbose:
            print(f"\nTraining {self.model_name}...")
            print(f"Training samples: {len(X_train)}")
            if X_val is not None:
                print(f"Validation samples: {len(X_val)}")

        if self.model is None:
            self.model = self._build_model()

        start_time = datetime.now()

        train_metrics = self._fit(X_train, y_train, X_val, y_val, verbose)

        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()

        self.is_trained = True
        self.classes_ = np.unique(y_train)
        self.training_history['last_trained'] = end_time.isoformat()
        self.training_history['training_time'] = training_duration
        self.training_history['train_metrics'] = train_metrics

        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val, dataset_name='validation', verbose=verbose)
            self.training_history['val_metrics'] = val_metrics

        if verbose:
            print(f"\nTraining completed in {training_duration:.2f} seconds")
            print(f"Training accuracy: {train_metrics.get('accuracy', 0):.4f}")
            if X_val is not None:
                val_acc = self.training_history['val_metrics'].get('accuracy', 0)
                print(f"Validation accuracy: {val_acc:.4f}")
                gap = train_metrics.get('accuracy', 0) - val_acc
                if gap > 0.05:
                    print(f"Train-val gap: {gap:.4f} (possible overfitting)")
                else:
                    print(f"Train-val gap: {gap:.4f} (good generalization)")

        return train_metrics

    @abstractmethod
    def _fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        pass

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_trained:
            raise ValueError(f"{self.model_name} has not been trained yet. Call train() first.")

        return self.model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_trained:
            raise ValueError(f"{self.model_name} has not been trained yet. Call train() first.")

        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError(f"{self.model_name} does not support probability predictions.")

        return self.model.predict_proba(X)

    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_true: Union[np.ndarray, pd.Series],
        dataset_name: str = 'test',
        verbose: bool = True
    ) -> Dict[str, float]:
        if not self.is_trained:
            raise ValueError(f"{self.model_name} has not been trained yet. Call train() first.")

        y_pred = self.predict(X)

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }

        if dataset_name == 'test':
            self.training_history['test_metrics'] = metrics

        if verbose:
            print(f"\n{dataset_name.capitalize()} Set Evaluation:")
            print(f"{'='*50}")
            print(f"Accuracy:           {metrics['accuracy']:.4f}")
            print(f"Precision (macro):  {metrics['precision_macro']:.4f}")
            print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
            print(f"Recall (macro):     {metrics['recall_macro']:.4f}")
            print(f"Recall (weighted):  {metrics['recall_weighted']:.4f}")
            print(f"F1-Score (macro):   {metrics['f1_macro']:.4f}")
            print(f"F1-Score (weighted): {metrics['f1_weighted']:.4f}")
            print(f"{'='*50}\n")

        return metrics

    def get_classification_report(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_true: Union[np.ndarray, pd.Series],
        output_dict: bool = False
    ) -> Union[str, Dict]:
        if not self.is_trained:
            raise ValueError(f"{self.model_name} has not been trained yet. Call train() first.")

        y_pred = self.predict(X)
        return classification_report(y_true, y_pred, output_dict=output_dict, zero_division=0)

    def get_confusion_matrix(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_true: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        if not self.is_trained:
            raise ValueError(f"{self.model_name} has not been trained yet. Call train() first.")

        y_pred = self.predict(X)
        return confusion_matrix(y_true, y_pred)

    def save(self, filepath: Union[str, Path], save_metadata: bool = True) -> None:
        if not self.is_trained:
            raise ValueError(f"{self.model_name} has not been trained yet. Cannot save untrained model.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'hyperparameters': self.hyperparameters,
            'classes_': self.classes_,
            'is_trained': self.is_trained,
            'training_history': self.training_history
        }

        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

        if save_metadata:
            metadata_path = filepath.parent / f"{filepath.stem}_metadata.json"
            metadata = {
                'model_name': self.model_name,
                'hyperparameters': self.hyperparameters,
                'training_history': self.training_history,
                'classes': self.classes_.tolist() if self.classes_ is not None else None
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata saved to {metadata_path}")

    def load(self, filepath: Union[str, Path]) -> None:
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.hyperparameters = model_data['hyperparameters']
        self.classes_ = model_data['classes_']
        self.is_trained = model_data['is_trained']
        self.training_history = model_data['training_history']

        print(f"Model loaded from {filepath}")

    def get_params(self) -> Dict[str, Any]:
        return self.hyperparameters.copy()

    def set_params(self, **params) -> 'BaseModel':
        self.hyperparameters.update(params)
        if self.model is not None:
            self.model = self._build_model()
            self.is_trained = False
        return self

    def get_training_history(self) -> Dict:
        return self.training_history.copy()

    def __repr__(self) -> str:
        trained_status = "trained" if self.is_trained else "not trained"
        return f"{self.model_name}({trained_status}, params={self.hyperparameters})"

    def __str__(self) -> str:
        return self.__repr__()
