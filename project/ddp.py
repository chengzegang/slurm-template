import os
from typing import Callable

import torch
import torch.distributed as dist


def start(func: Callable, kwargs: dict):
    if kwargs["ddp"]:
        if not kwargs['mnmg']:
            size = torch.cuda.device_count()
            torch.multiprocessing.spawn(
                ddp_setup,
                args=(func, kwargs),
                nprocs=size,
                join=True,
            )
        else:
            mnmg_ddp_setup(func, kwargs)
    else:
        func(kwargs)

def mnmg_ddp_setup(func: Callable, kwargs: dict):
    dist.init_process_group(backend="nccl")
    fn(**kwargs)
  
  
def ddp_setup(rank: int, world_size: int, func: Callable, kwargs: dict):
    """
    Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    backend = "nccl"
    if os.name == "nt":
        backend = "gloo"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
    else:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    fn(**kwargs)