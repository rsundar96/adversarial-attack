"""Module containing plotting functions."""

import matplotlib.pyplot as plt
import torch

from utils import MEAN, STANDARD_DEVIATION, denormalize


def plot_images(original_image: torch.Tensor, adversarial_image: torch.Tensor):
    """Plot the original and adversarial images.

    Args:
        original_image: Input image provided by the user.
        adversarial_image: Adversarial image generated from given input image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    original_image_denormalized = denormalize(
        original_image, mean=MEAN, std=STANDARD_DEVIATION
    )
    adversarial_image_denormalized = denormalize(
        adversarial_image, mean=MEAN, std=STANDARD_DEVIATION
    )

    # Original image
    original_image_np = original_image_denormalized.squeeze(0).permute(1, 2, 0).numpy()
    axes[0].imshow(original_image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Adversarial image
    adversarial_image_np = (
        adversarial_image_denormalized.squeeze(0).permute(1, 2, 0).detach().numpy()
    )
    axes[1].imshow(adversarial_image_np)
    axes[1].set_title("Adversarial Image")
    axes[1].axis("off")

    plt.show()
