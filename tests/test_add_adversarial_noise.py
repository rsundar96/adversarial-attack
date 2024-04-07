"""Tests for verifying the addition of adversarial noise to a given image."""

import pytest

from src.add_adversarial_noise import AdversarialImageGenerator, main, preprocess_image
from src.utils import load_image


@pytest.mark.parametrize(
    "image, target_class",
    [
        pytest.param(
            "images/panda.jpeg", "gibbon", id="Target class differs from input image"
        ),
        pytest.param(
            "images/panda.jpeg", "panda", id="Target class same as input image"
        ),
    ],
)
def test_e2e_adversarial_noise_attack(image, target_class):
    """End-to-end test that invokes the add_adversarial_noise script with an input img and target class.

    Args:
        image: Path to the input image.
        target_class: Target class to change model's prediction to.

    Asserts:
        Adversarial noise added to the input image changes model prediction to target class specified.
    """
    _, model_prediction = main(image, target_class)

    assert model_prediction == target_class


@pytest.mark.parametrize(
    "image, target_class", [pytest.param("images/panda.jpeg", "gibbon")]
)
def test_generate_adversarial_image_max_iterations_reached(image, target_class):
    """Test to check None type is returned for model prediction when max number of iterations is reached.

    Args:
        image: Path to the input image.
        target_class: Target class to change model's prediction to.

    Asserts:
        Model prediction is None when max number of iterations is reached.
    """
    input_img = load_image(image)
    input_img = preprocess_image(input_img)

    adversarial_img_generator = AdversarialImageGenerator(target_class)
    _, model_prediction = adversarial_img_generator.generate_adversarial_image(
        input_img, max_iterations=1
    )

    assert model_prediction is None
