"""Module containing code to add adversarial noise."""

import argparse
from utils import load_image, load_model


def main(input_img: str, target_class: int):
    _ = load_image(input_img)
    _ = load_model()

    print("Dummy statement")


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
