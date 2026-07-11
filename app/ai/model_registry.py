"""
This module implements a registry system for managing AI models.
"""

import uuid
from typing import Dict, Optional, List, Type

from app.ai.ai_model import AIModel
from app.core.logger import setup_logging

logger = setup_logging()

class ModelRegistry:
    """
    A singleton registry for managing AI models.
    Allows registering, retrieving, and listing AI models by their ID or name.
    """
    _instance: Optional["ModelRegistry"] = None
    _models: Dict[uuid.UUID, AIModel] = {}
    _name_to_id: Dict[str, uuid.UUID] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance

    def register_model(self, model: AIModel) -> None:
        """
        Registers an AI model with the registry.

        Args:
            model (AIModel): The AI model instance to register.

        Raises:
            ValueError: If a model with the same ID or name is already registered.
        """
        if model.model_id in self._models:
            logger.warning(f"Model with ID {model.model_id} already registered. Overwriting.")
        if model.model_name in self._name_to_id and self._name_to_id[model.model_name] != model.model_id:
            logger.error(f"Model with name '{model.model_name}' already registered with a different ID.")
            raise ValueError(f"Model with name '{model.model_name}' already registered.")

        self._models[model.model_id] = model
        self._name_to_id[model.model_name] = model.model_id
        logger.info(f"Registered AI model: {model.model_name} (ID: {model.model_id})")

    def unregister_model(self, model_id: uuid.UUID) -> None:
        """
        Unregisters an AI model from the registry by its ID.

        Args:
            model_id (uuid.UUID): The ID of the model to unregister.
        """
        if model_id in self._models:
            model = self._models.pop(model_id)
            if model.model_name in self._name_to_id and self._name_to_id[model.model_name] == model_id:
                del self._name_to_id[model.model_name]
            logger.info(f"Unregistered AI model: {model.model_name} (ID: {model_id})")
        else:
            logger.warning(f"Attempted to unregister non-existent model with ID: {model_id}")

    def get_model_by_id(self, model_id: uuid.UUID) -> Optional[AIModel]:
        """
        Retrieves an AI model by its ID.

        Args:
            model_id (uuid.UUID): The ID of the model to retrieve.

        Returns:
            Optional[AIModel]: The AI model instance, or None if not found.
        """
        return self._models.get(model_id)

    def get_model_by_name(self, model_name: str) -> Optional[AIModel]:
        """
        Retrieves an AI model by its name.

        Args:
            model_name (str): The name of the model to retrieve.

        Returns:
            Optional[AIModel]: The AI model instance, or None if not found.
        """
        model_id = self._name_to_id.get(model_name)
        if model_id:
            return self._models.get(model_id)
        return None

    def list_models(self) -> List[AIModel]:
        """
        Lists all registered AI models.

        Returns:
            List[AIModel]: A list of all registered AI model instances.
        """
        return list(self._models.values())

    def clear_registry(self) -> None:
        """
        Clears all registered models from the registry.
        """
        self._models.clear()
        self._name_to_id.clear()
        logger.info("AI model registry cleared.")

# Instantiate the singleton registry
model_registry = ModelRegistry()
