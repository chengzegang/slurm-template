import os
from typing import Callable

import torch
import torch.distributed as dist


def start(func: Callable, kwargs: dict):
    if kwargs["ddp"]:
        size = torch.cuda.device_count()
        torch.multiprocessing.spawn(
            init_process,
            args=(size, kwargs["port"], kwargs, func),
            nprocs=size,
            join=True,
        )
        cleanup()
    else:
        func(kwargs)


def init_process(rank: int, size: int, port: str, kwargs: dict, fn: Callable) -> None:
    backend = "nccl"
    if os.name == "nt":
        backend = "gloo"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
    else:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(**kwargs)


def cleanup(*args, **kwargs) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
