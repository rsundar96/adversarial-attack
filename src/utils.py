"""Module containing utility functions."""

import torch
import torchvision.models as models
from PIL import Image


def load_model() -> torch.nn.Module:
    """Loads a pretrained ResNet-18 model.

    Returns:
        Pretrained ResNet-18 model.
    """
    model = models.resnet18(pretrained=True)
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
