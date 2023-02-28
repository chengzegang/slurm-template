from __future__ import annotations
from typing import Any, Dict, Tuple
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import os
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed as dist


def save(
    model: nn.Module,
    ckptdir: str,
    steps: int,
    key_metric: float,
    larger_is_better: bool = False,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    prune: bool = False,
    mkdirs: bool = True,
    max_n: int = 5,
    appendix: Any | None = None,
) -> str:

    if not os.path.exists(ckptdir) and not mkdirs:
        raise ValueError(f"Directory {ckptdir} does not exist")

    if os.path.exists(ckptdir) and not os.path.isdir(ckptdir):
        raise ValueError(f"Path {ckptdir} is not a directory")

    os.makedirs(ckptdir, exist_ok=True)
    if prune and len(os.listdir(ckptdir)) >= max_n:
        ckptfiles = os.listdir(ckptdir)
        ckptfiles = [os.path.join(ckptdir, fn) for fn in ckptfiles]
        ckptfiles = [fn for fn in ckptfiles if fn.endswith(".pt")]
        ckptfiles = sorted(
            ckptfiles,
            key=lambda fn: fn.split(".")[0].split("_")[-1],
            reverse=larger_is_better,
        )
        ckptfiles = ckptfiles[:-max_n]
        for fn in ckptfiles:
            os.remove(fn)
    date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    key_metric_str = f"{key_metric:.2f}"
    ckptname = f"ckpt_{date}_{steps}_{key_metric_str}" + ".pt"
    ckptpath = os.path.join(ckptdir, ckptname)

    if isinstance(model, DistributedDataParallel):
        model = model.module
    if isinstance(optimizer, ZeroRedundancyOptimizer):
        optimizer.consolidate_state_dict(dist.get_rank())
    ckpt = dict(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict() if optimizer is not None else None,
        scheduler_state_dict=scheduler.state_dict() if scheduler is not None else None,
    )
    if appendix is not None:
        ckpt["appendix"] = appendix
    torch.save(ckpt, ckptpath)
    return ckptpath


def load(
    ckptpath: str,
) -> Dict:
    ckpt: Dict = torch.load(ckptpath)
    return ckpt


def loadon(
    ckptpath: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
) -> Tuple[nn.Module, Optimizer, LRScheduler]:
    ckpt = load(ckptpath)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return model, optimizer, scheduler
