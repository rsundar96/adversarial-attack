"""Module containing utility functions."""

import json
import os

import torch
import torchvision.models as models
from PIL import Image

IMAGENET_CLASS_IDX_VALUE_MAPPING = "imagenet_dataset/class_idx_value_mapping.json"


def get_imagenet_class_idx_from_value(class_value: str) -> int:
    """Gets the class index for a given class value from the ImageNet dataset.

    Args:
        class_value: The target class value to get the class index for.

    Returns:
        The class index for the given target class value.
    """
    IMAGENET_FILE_PATH = os.path.join(
        os.path.dirname(__file__), IMAGENET_CLASS_IDX_VALUE_MAPPING
    )
    class_idx = -1

    with open(IMAGENET_FILE_PATH, "r") as imagenet_class_idx_value_mapping_file:
        data = json.load(imagenet_class_idx_value_mapping_file)

    for key, value in data.items():
        if class_value in value:
            class_idx = key

    return int(class_idx)


def load_model() -> torch.nn.Module:
    """Loads a pretrained ResNet-18 model.

    Returns:
        Pretrained ResNet-18 model.
    """
    model = models.resnet18(weights="DEFAULT")
    model.eval()

    return model


def load_image(img_path: str) -> Image.Image:
    """Loads an image from a local path

    Args:
        img_path: Local path to an image to be read in.

    Returns:
        Loaded image.
    """
    image = Image.open(img_path)
    return image
