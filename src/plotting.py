"""Module containing plotting functions."""

import matplotlib.pyplot as plt
import torch

from schemas import Result
from utils import MEAN, STANDARD_DEVIATION, denormalize


def plot_images(orig_img: torch.Tensor, adv_img_results: Result):
    """Plot the original and adversarial images.

    Args:
        orig_img: Input image provided by the user.
        adv_img_results: Adversarial img results as well as original img label and probability score.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    original_image_denormalized = denormalize(
        orig_img, mean=MEAN, std=STANDARD_DEVIATION
    )
    adversarial_image_denormalized = denormalize(
        adv_img_results.adv_img, mean=MEAN, std=STANDARD_DEVIATION
    )

    # Original image
    original_image_np = original_image_denormalized.squeeze(0).permute(1, 2, 0).numpy()
    axes[0].imshow(original_image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[0].text(
        0.5,
        -0.08,
        f"Original Class: {adv_img_results.orig_label}\nProbability: {adv_img_results.orig_prob}%",
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[0].transAxes,
    )

    # Adversarial image
    adversarial_image_np = (
        adversarial_image_denormalized.squeeze(0).permute(1, 2, 0).detach().numpy()
    )
    axes[1].imshow(adversarial_image_np)
    axes[1].set_title("Adversarial Image")
    axes[1].axis("off")
    axes[0].text(
        1.7,
        -0.08,
        f"Target (Predicted) Class: {adv_img_results.adv_label}\nProbability: {adv_img_results.adv_prob}%",
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[0].transAxes,
    )

    plt.show()
