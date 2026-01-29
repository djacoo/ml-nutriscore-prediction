"""
Model registry for managing and instantiating different ML models.

This module provides a centralized registry system for all available models
in the project, making it easy to create, track, and manage model instances.
"""

from typing import Dict, Type, Optional, Any
from pathlib import Path
import json


class ModelRegistry:
    """
    Registry for managing available machine learning models.

    This class maintains a central registry of all model classes and provides
    utilities for instantiating models, listing available models, and managing
    model metadata.

    Attributes:
        _models: Dictionary mapping model names to model classes
        _metadata: Dictionary storing metadata about registered models
    """

    _models: Dict[str, Type] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        model_class: Type,
        description: str = "",
        category: str = "general",
        requires_probability: bool = False
    ) -> Type:
        """
        Register a model class in the registry.

        Args:
            name: Unique name for the model (e.g., 'logistic_regression')
            model_class: The model class to register
            description: Brief description of the model
            category: Category of the model (e.g., 'linear', 'tree-based', 'ensemble')
            requires_probability: Whether the model requires probability estimates

        Returns:
            The registered model class (for use as decorator)

        Raises:
            ValueError: If a model with the same name is already registered
        """
        if name in cls._models:
            raise ValueError(f"Model '{name}' is already registered.")

        cls._models[name] = model_class
        cls._metadata[name] = {
            'description': description,
            'category': category,
            'requires_probability': requires_probability,
            'class_name': model_class.__name__
        }

        return model_class

    @classmethod
    def get_model_class(cls, name: str) -> Type:
        """
        Get a model class by name.

        Args:
            name: Name of the model to retrieve

        Returns:
            The model class

        Raises:
            KeyError: If model name is not registered
        """
        if name not in cls._models:
            available = ', '.join(cls._models.keys())
            raise KeyError(
                f"Model '{name}' not found in registry. "
                f"Available models: {available}"
            )

        return cls._models[name]

    @classmethod
    def create_model(cls, name: str, **hyperparameters) -> Any:
        """
        Create an instance of a registered model.

        Args:
            name: Name of the model to create
            **hyperparameters: Hyperparameters to pass to the model

        Returns:
            Instance of the requested model

        Raises:
            KeyError: If model name is not registered
        """
        model_class = cls.get_model_class(name)
        return model_class(**hyperparameters)

    @classmethod
    def list_models(cls) -> Dict[str, Dict[str, Any]]:
        """
        List all registered models with their metadata.

        Returns:
            Dictionary mapping model names to their metadata
        """
        return cls._metadata.copy()

    @classmethod
    def get_models_by_category(cls, category: str) -> Dict[str, Type]:
        """
        Get all models in a specific category.

        Args:
            category: Category to filter by

        Returns:
            Dictionary mapping model names to model classes for the category
        """
        return {
            name: cls._models[name]
            for name, metadata in cls._metadata.items()
            if metadata['category'] == category
        }

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a model is registered.

        Args:
            name: Name of the model to check

        Returns:
            True if model is registered, False otherwise
        """
        return name in cls._models

    @classmethod
    def get_model_info(cls, name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata information about a registered model.

        Args:
            name: Name of the model

        Returns:
            Dictionary containing model metadata, or None if not found
        """
        return cls._metadata.get(name)

    @classmethod
    def print_registry(cls) -> None:
        """Print a formatted list of all registered models."""
        if not cls._models:
            print("No models registered yet.")
            return

        print("\n" + "="*70)
        print("REGISTERED MODELS")
        print("="*70)

        # Group by category
        categories = {}
        for name, metadata in cls._metadata.items():
            category = metadata['category']
            if category not in categories:
                categories[category] = []
            categories[category].append((name, metadata))

        for category, models in sorted(categories.items()):
            print(f"\n{category.upper()} MODELS:")
            print("-"*70)
            for name, metadata in sorted(models):
                prob_req = "✓" if metadata['requires_probability'] else "✗"
                print(f"  • {name:25} | Prob: {prob_req} | {metadata['description']}")

        print("\n" + "="*70 + "\n")

    @classmethod
    def save_registry_info(cls, filepath: Path) -> None:
        """
        Save registry information to a JSON file.

        Args:
            filepath: Path where to save the registry info
        """
        registry_data = {
            'models': cls._metadata,
            'total_models': len(cls._models)
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(registry_data, f, indent=2)

        print(f"Registry information saved to {filepath}")


def register_model(
    name: str,
    description: str = "",
    category: str = "general",
    requires_probability: bool = False
):
    """
    Decorator for registering model classes.

    This is a convenience decorator that can be used to automatically
    register model classes when they are defined.

    Args:
        name: Unique name for the model
        description: Brief description of the model
        category: Category of the model
        requires_probability: Whether the model requires probability estimates

    Returns:
        Decorator function

    Example:
        @register_model(
            name='logistic_regression',
            description='Logistic Regression baseline model',
            category='linear'
        )
        class LogisticRegressionModel(BaseModel):
            ...
    """
    def decorator(model_class: Type) -> Type:
        ModelRegistry.register(
            name=name,
            model_class=model_class,
            description=description,
            category=category,
            requires_probability=requires_probability
        )
        return model_class

    return decorator
