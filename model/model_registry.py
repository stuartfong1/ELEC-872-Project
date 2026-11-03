from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch
from torch import nn
from torchvision import models, transforms

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class RetrievedModel:
    """Result of retrieval from the model registry.

    Parameters
    ----------
    model : nn.Module
        Embedding model
    transform : Callable
        Preprocessing applied to each PIL image before it is passed to the model

    """

    model: nn.Module
    transform: Callable


class ModelRegistry:
    """Load models and their transforms by name, caching each one after first use.

    Call set_config once with the training config before calling get_model.
    """

    _models: ClassVar[dict[str, RetrievedModel]] = {}
    _config: ClassVar[dict | None] = None

    @classmethod
    def set_config(cls, config: dict) -> None:
        """Set the config used to build the models.

        Parameters
        ----------
        config : dict
            Settings from the training YAML config

        """
        cls._config = config

    @classmethod
    def get_model(cls, model_name: str) -> RetrievedModel:
        """Retrieve the model if it has already been built, otherwise build it.

        Parameters
        ----------
        model_name : str
            Name of the model to retrieve

        Returns
        -------
        RetrievedModel
            Model and image transform

        """
        if model_name in cls._models:
            return cls._models[model_name]

        match model_name:
            case "resnet50":
                # ResNet50 trained from scratch
                # Set num_classes to the embedding dimension
                # so that it outputs the embeddings directly
                model = models.resnet50(weights=None, num_classes=cls._config["embedding_dim"])

                # Optionally, load a checkpoint saved by train.py
                # Checkpoints store the model weights and optimizer state
                if cls._config["weights_path"] is not None:
                    checkpoint = torch.load(cls._config["weights_path"], map_location="cpu")
                    model.load_state_dict(checkpoint["model"])

                # Resize the shorter side to 224 and center-crop to 224x224
                # Normalize using the ImageNet mean/std
                transform = transforms.Compose(
                    [
                        transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225),
                        ),
                    ],
                )

                cls._models[model_name] = RetrievedModel(
                    model=model,
                    transform=transform,
                )

        return cls._models[model_name]
