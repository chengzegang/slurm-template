from __future__ import annotations

from functools import partial
from typing import Tuple

import torch
from torch import nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR

from . import lr_lambda


def get_optimizer(
    model: nn.Module,
    optimizer: str,
    lr: float,
    weight_decay: float,
    betas: Tuple[float, float],
    momentum: float,
    eps: float,
    **kwargs,
) -> torch.optim.Optimizer:
    if optimizer == "adam":
        return Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
    elif optimizer == "sgd":
        return SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    elif optimizer == "adamw":
        return AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
    else:
        raise NotImplementedError(f"Optimizer {optimizer} not implemented")


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler: str,
    total_steps,
    warmup_steps,
    start_steps,
    **kwargs,
) -> LambdaLR:
    if scheduler == "cosine":
        return LambdaLR(
            optimizer,
            partial(lr_lambda.cosine_lr_lambda, start_steps, total_steps),
        )
    elif scheduler == "cosine_warmup":
        return LambdaLR(
            optimizer,
            partial(
                lr_lambda.cosine_warmup_lr_lambda,
                start_steps,
                warmup_steps,
                total_steps,
            ),
        )
    elif scheduler == "identity":
        return LambdaLR(optimizer, lr_lambda.identity_lr_lambda)
    else:
        raise NotImplementedError(f"Scheduler {scheduler} not implemented")


def get_optims(
    model: nn.Module,
    optimizer: str,
    lr: float,
    weight_decay: float,
    betas: Tuple[float, float],
    momentum: float,
    eps: float,
    scheduler: str,
    start_steps,
    warmup_steps,
    total_steps,
    **kwargs,
) -> Tuple[torch.optim.Optimizer, LambdaLR]:

    optimizer = get_optimizer(
        model, optimizer, lr, weight_decay, betas, momentum, eps, **kwargs
    )

    scheduler = get_scheduler(
        optimizer, scheduler, total_steps, warmup_steps, start_steps, **kwargs
    )

    return optimizer, scheduler
