# Adversarial Noise

This repository contains code that adds adversarial noise to a given image, tricking an image classification model into misclassifying the altered image as a specified target class.

## 🔨 Requirements and Installation

[Miniforge (Conda)](https://github.com/conda-forge/miniforge) is recommended for creating a Python environment.

```shell
# Clone the repo
git clone https://github.com/rsundar96/adversarial-attack.git

# Create a Python environment with all of the required dependencies
conda env create -f environment.yml -v

# (For local development only)
# Setup pre-commit for code formatting
pre-commit install

# Activate the environment
conda activate adversarial-attack
```

## 🧑‍💻 Local Development

[Black](https://black.readthedocs.io/en/stable) formatting should be applied on all code before making a PR - if properly setup, this is covered by pre-commit.

When updating dependencies, the `requirements.txt` file also needs to be updated in addition to `pyproject.toml`. The following command should update the `requirements.txt` file.

```shell
pip-compile --extra=dev --output-file=requirements.txt pyproject.toml
```

### 🧪 Tests

Test framework in use is [pytest](https://docs.pytest.org/). To check code coverage of tests, run the following commands

```shell
# Run tests using code coverage
coverage run --module pytest tests

# Generate test report
coverage report -m -i
coverage html -i
```

## 🚀 Usage

A few sample images are present under the [images](images) folder. You may use these images, or upload images into this folder.

The list of target classes currently supported are those present in the [ImageNet dataset](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/).

```shell
# Example to make the pre-trained ResNet-18 image classification model misclassify a panda as a gibbon
python src/add_adversarial_noise.py --image images/panda.jpeg --target-class "gibbon"
```
