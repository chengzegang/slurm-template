from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from .. import models
from ..datasets import ImageFolder
from ..transforms import AddNoise, ResizeCenterCrop
from .utils import checkpoint
from .base_trainer import Trainer
from .utils.logger import Logger, TensorboardLogger
from .utils.optims import get_optims
from .utils.runs import get_run_dir

if torch.__version__.startswith("2"):
    from torch.optim.lr_scheduler import LRScheduler  # type: ignore
else:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


class VitDenoiseTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        transforms: nn.Module,
        augmentations: nn.Module,
        root: str,
        logdir: str,
        optimizer: str,
        scheduler: str,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float],
        momentum: float,
        eps: float,
        total_epochs: int,
        warmup_epochs: int,
        log_times_per_epoch: int,
        ckpt_times_per_epoch: int,
        device: str,
        verbose: bool,
        **kwargs,
    ):
        self.slurm_proc_id = os.environ.get("SLURM_PROCID", None)
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        print(f'process started with {self.local_rank}, {self.global_rank}, {self.world_size}')
        self.device = device
        self.model = model
        if dist.is_available() and dist.is_initialized() and self.device != "cpu":
            self.device = f"cuda:{self.local_rank}"
            self.model = DDP(self.model.to(self.device), device_ids=[self.local_rank])
        print("model initialized")

        self.logdir: str | None = None
        self.ckptdir: str | None = None
        self.logger: Logger | None = None
        self.verbose = False

        if self.local_rank == 0:
            self.logdir = get_run_dir(logdir)
            self.ckptdir = os.path.join(self.logdir, "ckpts")
            os.makedirs(self.ckptdir, exist_ok=True)
            self.logger = TensorboardLogger(self.logdir)
            self.verbose = verbose
        self.transforms = transforms
        self.dataset = self._create_dataset(root, self.transforms, self.verbose)
        self.sampler: Sampler | None = None
        if dist.is_available() and dist.is_initialized():
            self.sampler = DistributedSampler(self.dataset, shuffle=shuffle)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False if self.sampler is not None else shuffle,
            num_workers=num_workers,
            sampler=self.sampler,
        )
        print("dataloader initialized")

        self.total_epochs = total_epochs
        self.steps_per_epoch = len(self.dataloader)
        self.total_steps = total_epochs * self.steps_per_epoch
        self.warmup_steps = warmup_epochs * self.steps_per_epoch
        self.optimizer, self.scheduler = get_optims(
            self.model,
            optimizer,
            lr,
            weight_decay,
            betas,
            momentum,
            eps,
            scheduler,
            0,
            self.warmup_steps,
            self.total_steps,
            **kwargs,
        )
        print("optimizer initialized")
        self.log_per_steps = self.steps_per_epoch // log_times_per_epoch
        self.ckpt_per_steps = self.steps_per_epoch // ckpt_times_per_epoch

        self.augmentations = augmentations

    @classmethod
    def _create_dataset(cls, root: str, transforms: Callable, verbose: bool) -> Dataset:
        return ImageFolder(root, transforms=transforms, verbose=verbose)

    @classmethod
    def _tensor_to_img(cls, x: torch.Tensor, reverse_norm: bool = False) -> np.ndarray:
        img: np.ndarray = x.detach().cpu().permute(-2, -1, -3).numpy().astype(np.uint8)
        return img

    @classmethod
    def _viz(cls, x: torch.Tensor, y: torch.Tensor, yh: torch.Tensor):
        x_img = cls._tensor_to_img(x[0], reverse_norm=True)
        y_img = cls._tensor_to_img(y[0])
        yh_img = cls._tensor_to_img(yh[0])
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(x_img)
        axes[0].set_title("Input")
        axes[1].imshow(y_img)
        axes[1].set_title("Target")
        axes[2].imshow(yh_img)
        axes[2].set_title("Prediction")
        for ax in axes:
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        return fig

    @classmethod
    def _train_step(
        cls,
        x: torch.Tensor,
        y: torch.Tensor,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LambdaLR | None = None,
        device: str | torch.device = "cpu",
    ) -> tuple[
        nn.Module, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float
    ]:
        x = x.to(device)
        y = y.to(device)
        yh = model(x)
        optimizer.zero_grad()
        loss = F.mse_loss(y, yh)
        loss.backward()
        optimizer.step()
        curr_lr = -1.0
        if scheduler is not None:
            scheduler.step()
            curr_lr = scheduler.get_last_lr()[0]
        return model, x, y, yh, loss, curr_lr

    @classmethod
    def _train(
        cls,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: Optimizer,
        scheduler: LambdaLR | None = None,
        sampler: Sampler | None = None,
        logger: Logger | None = None,
        total_epochs: int = 10,
        log_per_steps: int = 10,
        ckpt_per_steps: int = 100,
        ckptdir: str | None = None,
        augmentations: nn.Module | None = None,
        device: str = "cpu",
        global_rank: int = 0,
        local_rank: int = 0,
    ):
        steps = 0
        data_used = 0
        model.train()
        model.to(device)
        pbar = tqdm(range(total_epochs * len(dataloader)))
        for epoch in range(total_epochs):
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)
                print(
                    f"[GPU{global_rank}:{local_rank}] Epoch {epoch} | Batchsize {dataloader.batch_size} | Steps {len(dataloader)}"
                )
            for idx, x in enumerate(dataloader):
                if x.dim() == 3:
                    x = x.unsqueeze(0)
                y = x.detach().clone()
                y0 = y.detach().clone().cpu()
                if augmentations is not None:
                    x = augmentations(x)
                x0 = x.detach().clone().cpu()
                results = cls._train_step(x, y, model, optimizer, scheduler, device)
                fig = cls._viz(x0, y0, results[3])
                if logger is not None and steps % log_per_steps:
                    logger.add_scalar("train/loss", results[4], data_used)
                    logger.add_scalar("train/lr", results[5], data_used)
                    logger.add_figure("train/imgs", fig, data_used)
                if ckptdir is not None and steps % ckpt_per_steps:
                    metrics = dict(
                        epoch=epoch,
                        steps=steps,
                        data_used=data_used,
                        lr=results[5],
                        loss=results[4].item(),
                    )

                    checkpoint.save(
                        model,
                        ckptdir,
                        steps,
                        results[4].item(),
                        False,
                        optimizer,
                        scheduler,
                        True,
                        appendix=metrics,
                    )
                steps += 1
                pbar.set_postfix(loss=results[4].item(), lr=results[5])
                pbar.update(1)
        return model

    def train(self):
        return self._train(
            self.model,
            self.dataloader,
            self.optimizer,
            self.scheduler,
            self.sampler,
            self.logger,
            self.total_epochs,
            self.log_per_steps,
            self.ckpt_per_steps,
            self.ckptdir,
            self.augmentations,
            self.device,
            self.global_rank,
            self.local_rank,
        )


def build(**kwargs):
    model = models.build(**kwargs)
    del kwargs["model"]
    transforms = ResizeCenterCrop(kwargs["image_size"] + 32, kwargs["image_size"])
    augmentations = AddNoise()
    trainer = VitDenoiseTrainer(model, transforms, augmentations, **kwargs)
    return trainer
