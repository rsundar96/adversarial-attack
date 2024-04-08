"""Module for defining schemas."""

from dataclasses import dataclass

import torch


@dataclass
class Result:
    """Dataclass containing info for plotting purposes."""

    adv_img: torch.Tensor = None
    adv_label: int = None
    adv_prob: float = None
    orig_label: str = None
    orig_prob: float = None
