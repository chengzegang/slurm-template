import os
from functools import wraps
from typing import Callable, Iterable

import torch
import torch.distributed as dist


def _func_wrapper(func):
    @wraps(func)
    def _wrapper(*args, **kwargs):
        torch.cuda.set_device(dist.get_rank())
        return func(*args, **kwargs)

    return _wrapper


def start(func: Callable, args: dict):
    if args["ddp"]:
        func = _func_wrapper(func)
        size = torch.cuda.device_count()
        torch.multiprocessing.spawn(
            init_process,
            args=(size, args["port"], args, func),
            nprocs=size,
            join=True,
        )
        cleanup()
    else:
        func(args, rank=0, size=1)


def init_process(rank: int, size: int, port: str, args: Iterable, fn: Callable) -> None:
    backend = "nccl"
    if os.name == "nt":
        backend = "gloo"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
    else:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(args, rank, size)


def cleanup(*args, **kwargs) -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
