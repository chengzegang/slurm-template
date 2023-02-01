from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.distributed as dist
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class Logger(ABC):
    @abstractmethod
    def add_scalar(
        self, tag: str, value: int | float | torch.Tensor, step: int
    ) -> None:
        pass

    @abstractmethod
    def add_figure(self, tag: str, figure: plt.figure, step: int) -> None:
        pass


class TensorboardLogger(Logger):
    def __init__(self, logdir: str) -> None:
        super().__init__()
        self._writter = self._get_logger(logdir)

    def _get_logger(self, logdir: str) -> SummaryWriter:
        runs_dir = os.path.join(logdir, "runs")
        os.makedirs(runs_dir, exist_ok=True)
        return SummaryWriter(runs_dir)

    def add_scalar(self, tag: str, scalar: int | float | torch.Tensor, number: int):
        self._writter.add_scalar(tag, scalar, number)

    def add_figure(self, tag: str, scalar: int | float | torch.Tensor, number: int):
        self._writter.add_figure(tag, scalar, number)
