"""Module containing code to add adversarial noise."""

import argparse
import logging
from typing import Tuple

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from plotting import plot_images
from schemas import Result
from utils import (
    MEAN,
    STANDARD_DEVIATION,
    get_class_values_from_idx,
    get_imagenet_class_idx_from_value,
    load_image,
    load_model,
)


class AdversarialImageGenerator:
    """Generates adversarial images using a pre-trained model that matches the target class specified by the user."""

    def __init__(self, target_class: str):
        """Initializes the instance using a pre-trained ResNet18 model and the target class.

        Args:
            model: Pre-trained ResNet18 model.
            target_class: Target class specified by the user.
        """
        self.model = load_model()
        self.target_class = target_class

    def calculate_probability(self, output: torch.Tensor, prediction: int) -> float:
        """Calculates the probability, given the model's prediction and output.

        Args:
            output: Scores predicted by the model for each class.
            prediction: Class predicted by the model.

        Returns:
            Probability score associated with given class predicted by the model, in the form of percentage.
        """
        probability = torch.softmax(output, dim=1)[0, prediction].item()
        probability = float("{:.2f}".format(probability * 100))

        label = get_class_values_from_idx(prediction)

        return probability, label

    def model_prediction(self, img: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Generates a prediction using a pre-trained model.

        Args:
            img: Image to make the prediction on.

        Returns:
            A tuple containing the output tensor and the predicted class' index
        """
        output = self.model(img)
        _, prediction = torch.max(output, 1)

        return output, prediction.item()

    def iterative_target_class_method(
        self,
        input_img: torch.Tensor,
        data_grad: torch.Tensor,
        epsilon: float = 0.25,
        alpha: float = 0.025,
    ) -> torch.Tensor:
        """Implements the iterative target class method by removing perturbation from the input image.

        Args:
            input_img: Input image to perform the target class method on.
            data_grad: Gradient of the loss with respect to input image.
            epsilon: Maximum allowed perturbation in each pixel.
            alpha: Step size used to compute the applied perturbation.

        Returns:
            Image Tensor with adversarial noise added.
        """
        x_adv = input_img - (alpha * data_grad.sign())
        total_grad = x_adv - input_img
        total_grad = torch.clamp(total_grad, -epsilon, epsilon)
        adversarial_img = input_img + total_grad

        return adversarial_img

    def generate_adversarial_image(
        self, input_img: torch.Tensor, max_iterations: int = 5
    ) -> Result:
        """Generates an adversarial image using the Iterative Target Class Method.

        Args:
            input_img: Input image to perform the target class method on.
            max_iterations: Maximum number of iterations to run the target class method.

        Returns:
            A Result dataclass containing the adversarial img, label and probability score, along with the original image's label and probability score.
        """
        input_img = input_img.detach().requires_grad_(True)

        # Get expected target class idx
        expected_adversarial_class = get_imagenet_class_idx_from_value(
            self.target_class
        )

        # Compute model probability on original input img
        orig_model_output, orig_model_prediction = self.model_prediction(input_img)
        clean_prob, clean_label = self.calculate_probability(
            orig_model_output, orig_model_prediction
        )

        for _ in tqdm(
            range(max_iterations),
            desc="Adding adversarial noise to image to match target class",
        ):
            model_output, model_prediction = self.model_prediction(input_img)

            # Calculate loss
            loss = torch.nn.CrossEntropyLoss()
            loss_value = loss(model_output, torch.tensor([expected_adversarial_class]))

            # Compute gradients
            self.model.zero_grad()
            loss_value.backward()
            data_grad = input_img.grad.data

            adversarial_img = self.iterative_target_class_method(input_img, data_grad)
            adversarial_img = adversarial_img.detach().requires_grad_(True)

            _, adversarial_img_prediction = self.model_prediction(adversarial_img)

            input_img = adversarial_img

        if adversarial_img_prediction == expected_adversarial_class:
            logging.info(
                "Adversarial image has been generated and is classified as target class."
            )
            adv_prob, adv_label = self.calculate_probability(
                model_output, adversarial_img_prediction
            )
            return Result(
                adv_img=adversarial_img,
                adv_label=adv_label,
                adv_prob=adv_prob,
                orig_label=clean_label,
                orig_prob=clean_prob,
            )
        else:
            logging.warning(
                "Maximum number of iterations reached. Adversarial image has not been generated to match target class."
            )
            return Result()


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocesses the image by applying a series of transformations before feeding it to the model.

    Args:
        image: Image to be preprocessed.

    Returns:
        Preprocessed image with dimensions (B, C, H, W)
    """
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STANDARD_DEVIATION),
        ]
    )
    input_tensor = preprocess(image)
    input_tensor = input_tensor.unsqueeze(0)

    return input_tensor


def main(input_image: str, target_class: str):
    """Runner function for adding adversarial noise."""
    input_image = load_image(input_image)
    input_image = preprocess_image(input_image)

    adversarial_img_generator = AdversarialImageGenerator(target_class)

    adv_img_results = adversarial_img_generator.generate_adversarial_image(input_image)

    plot_images(input_image, adv_img_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image", type=str, help="Input image to add adversarial noise to."
    )
    parser.add_argument(
        "--target-class", type=str, help="Target class of the input image."
    )

    args = parser.parse_args()

    main(args.image, args.target_class)
