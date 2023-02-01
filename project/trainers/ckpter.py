from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Tuple

import torch
from torch.distributed.optim import ZeroRedundancyOptimizer as ZRO
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

if torch.__version__.startswith("2"):
    from torch.optim.lr_scheduler import LRScheduler  # type: ignore
else:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


def save(
    ckpt_path: str,
    model: Module | None = None,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    metrics: Dict | None = None,
    mkdirs: bool = False,
    overwrite: bool = False,
    **kwargs,
) -> None:

    if os.path.exists(ckpt_path) and not overwrite:
        raise ValueError(f"Checkpoint already exists at {ckpt_path}.")

    if mkdirs:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    if isinstance(model, DDP):
        model = model.module

    if isinstance(optimizer, ZRO):
        optimizer.consolidate_state_dict(to=0)

    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    torch.save(
        dict(
            date=date,
            model_state_dict=model.state_dict() if model is not None else None,
            optimizer_state_dict=optimizer.state_dict()
            if optimizer is not None
            else None,
            scheduler_state_dict=scheduler.state_dict()
            if scheduler is not None
            else None,
            metrics=metrics if metrics is not None else None,
        ),
        ckpt_path,
    )


def load(
    ckpt_path: str,
    model: Module | None = None,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    metrics: Dict | None = None,
    **kwargs,
) -> Tuple[Module | None, Optimizer | None, LRScheduler | None, Dict | None, str]:

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"]) if model is not None else None
    optimizer.load_state_dict(
        ckpt["optimizer_state_dict"]
    ) if optimizer is not None else None
    scheduler.load_state_dict(
        ckpt["scheduler_state_dict"]
    ) if scheduler is not None else None
    metrics.update(ckpt["metrics"]) if metrics is not None else None

    date = ckpt["date"]

    return model, optimizer, scheduler, metrics, date
