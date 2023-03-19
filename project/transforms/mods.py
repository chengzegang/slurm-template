from __future__ import annotations

from typing import Tuple

import kornia as K
import torch
import torchvision.transforms.functional as TF
from torch import nn


class ResizeCenterCrop(nn.Module):
    def __init__(
        self, resize_size: int | Tuple[int, int], crop_size: int | Tuple[int, int]
    ) -> None:
        super().__init__()

        self.transforms = nn.Sequential(
            K.augmentation.Resize(resize_size),
            K.augmentation.CenterCrop(crop_size),
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float()
        x = self.transforms(x)
        return x  # type: ignore


class AddNoise(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.transforms = nn.Sequential(
            K.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.9),
            K.augmentation.RandomPlasmaShadow(p=0.9),
            K.augmentation.RandomGaussianNoise(0, std=3, p=0.9),
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float()
        x = self.transforms(x)
        return x  # type: ignore


class Normalize(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.normlize = K.enhance.Normalize(mean=[0, 0, 0], std=[1, 1, 1])

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float()
        x = self.normlize(x)
        return x  # type: ignore


class ReverseNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def _inverse_normalize(cls, x: torch.Tensor) -> torch.Tensor:
        x = TF.normalize(
            x,
            [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            [1 / 0.229, 1 / 0.224, 1 / 0.225],
        )
        x = x.clamp(0, 1)
        return x  # type: ignore

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float()
        x = self._inverse_normalize(x)
        x = x * 255
        x = x.to(torch.uint8)
        return x
