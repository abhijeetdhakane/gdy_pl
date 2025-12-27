"""
Model Registry for GDyNet Models

This module provides a centralized registry for mapping model names to their
implementations and compatible dataset classes. It ensures proper pairing of
models with their required datasets.

Usage:
    from config.registry import ModelRegistry

    # Get model and dataloader classes
    model_class = ModelRegistry.get_model_class('gdynet_vanilla')
    dataloader_class = ModelRegistry.get_dataloader_class('gdynet_vanilla')

    # Check if model is registered
    if ModelRegistry.is_registered('gdynet_ferro'):
        info = ModelRegistry.get_model_info('gdynet_ferro')
"""

from typing import Dict, Any
from importlib import import_module


class ModelRegistry:
    """Registry for mapping model names to their implementations and datasets."""

    _REGISTRY = {
        'gdynet_vanilla': {
            'model_module': 'models.gdynet_vanilla',
            'model_class': 'CrystalGraphConvNet',
            'dataloader_module': 'data.gdynet_dataloader',
            'dataloader_class': 'PyGMDStackGen_vanilla',
            'description': 'Vanilla GDyNet without atom direction features',
            'requires_direction': False,
            'required_files': ['atom_types', 'target_index', 'nbr_lists', 'nbr_dists'],
        },
        'gdynet_ferro': {
            'model_module': 'models.gdynet_ferro',
            'model_class': 'CrystalGraphConvNet',
            'dataloader_module': 'data.gdynet_dataloader',
            'dataloader_class': 'PyGMDStackGen_ferro',
            'description': 'GDyNet with atom direction features for ferroelectric materials',
            'requires_direction': True,
            'required_files': ['atom_types', 'target_index', 'nbr_lists', 'nbr_dists', 'atom_directions'],
        },
    }

    @classmethod
    def is_registered(cls, model_name: str) -> bool:
        """
        Check if a model is registered.

        Args:
            model_name: Name of the model to check

        Returns:
            True if model is registered, False otherwise
        """
        return model_name in cls._REGISTRY

    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """
        Get complete model information.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary containing model information

        Raises:
            ValueError: If model is not registered
        """
        if not cls.is_registered(model_name):
            raise ValueError(
                f"Model '{model_name}' not registered. "
                f"Available models: {list(cls._REGISTRY.keys())}"
            )
        return cls._REGISTRY[model_name].copy()

    @classmethod
    def get_model_class(cls, model_name: str):
        """
        Get model class dynamically.

        Args:
            model_name: Name of the model

        Returns:
            Model class (uninstantiated)

        Raises:
            ValueError: If model is not registered
            ImportError: If model module cannot be imported
        """
        info = cls.get_model_info(model_name)
        module = import_module(info['model_module'])
        return getattr(module, info['model_class'])

    @classmethod
    def get_dataloader_class(cls, model_name: str):
        """
        Get dataloader class dynamically.

        Args:
            model_name: Name of the model

        Returns:
            Dataloader class (uninstantiated)

        Raises:
            ValueError: If model is not registered
            ImportError: If dataloader module cannot be imported
        """
        info = cls.get_model_info(model_name)
        module = import_module(info['dataloader_module'])
        return getattr(module, info['dataloader_class'])

    @classmethod
    def list_models(cls) -> Dict[str, str]:
        """
        List all registered models with descriptions.

        Returns:
            Dictionary mapping model names to descriptions
        """
        return {name: info['description'] for name, info in cls._REGISTRY.items()}

    @classmethod
    def get_required_files(cls, model_name: str) -> list:
        """
        Get list of required data files for a model.

        Args:
            model_name: Name of the model

        Returns:
            List of required file names (without .npy extension)

        Raises:
            ValueError: If model is not registered
        """
        info = cls.get_model_info(model_name)
        return info['required_files']

    @classmethod
    def requires_direction_features(cls, model_name: str) -> bool:
        """
        Check if model requires atom direction features.

        Args:
            model_name: Name of the model

        Returns:
            True if model requires direction features, False otherwise

        Raises:
            ValueError: If model is not registered
        """
        info = cls.get_model_info(model_name)
        return info['requires_direction']
