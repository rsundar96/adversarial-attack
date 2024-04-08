"""Tests for verifying the addition of adversarial noise to a given image."""

import pytest

from src.add_adversarial_noise import AdversarialImageGenerator, preprocess_image
from src.utils import load_image


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
    results = adversarial_img_generator.generate_adversarial_image(
        input_img, max_iterations=1
    )

    assert results.adv_img is None


@pytest.mark.parametrize(
    "image, target_class", [pytest.param("images/tabby_cat.jpeg", "Roger Federer")]
)
def test_generate_adversarial_image_incorrect_class_value(image, target_class):
    """Test to check a SystemExit occurs for incorrect class value.

    Args:
        image: Path to the input image.
        target_class: Target class to change model's prediction to.
    """
    input_img = load_image(image)
    input_img = preprocess_image(input_img)

    adversarial_img_generator = AdversarialImageGenerator(target_class)

    with pytest.raises(SystemExit) as exc_info:
        adversarial_img_generator.generate_adversarial_image(input_img)

    assert exc_info.value.code == 1


@pytest.mark.parametrize(
    "image, target_class, original_class",
    [
        pytest.param("images/panda.jpeg", "gibbon", "panda"),
        pytest.param("images/tabby_cat.jpeg", "ostrich", "tabby cat"),
    ],
)
def test_generate_adversarial_image(image, target_class, original_class):
    """Test to check adversarial image results are generated and match the target class

    Args:
        image: Path to the input image.
        target_class: Target class to change model's prediction to.

    Asserts:
        Target and Original class values are present in the obtained results.
    """
    input_img = load_image(image)
    input_img = preprocess_image(input_img)

    adversarial_img_generator = AdversarialImageGenerator(target_class)
    results = adversarial_img_generator.generate_adversarial_image(input_img)

    assert target_class in results.adv_label
    assert original_class in results.orig_label
