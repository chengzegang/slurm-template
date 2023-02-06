import shutil
from typing import Tuple

import click

from . import ddp, trainers


@click.group(name="project")
def cli():
    pass


@cli.command("train")
@click.option("--ddp", is_flag=True, help="Use DDP")
@click.option("--trainer", "-t", type=str, required=True, help="Trainer name")
@click.option("--model", "-m", type=str, required=True, help="Model name")
@click.option("--root", "-r", type=str, required=True, help="Path to dataset")
@click.option("--logdir", type=str, help="Path to logdir", default="logs")
@click.option("--optimizer", type=str, help="Optimizer", default="adamw")
@click.option("--scheduler", type=str, help="Scheduler", default="cosine")
@click.option("--batch-size", type=int, help="Batch size", default=1)
@click.option("--num-workers", type=int, help="Number of workers", default=0)
@click.option("--shuffle", type=bool, help="Shuffle", default=True)
@click.option("--lr", type=float, help="Learning rate", default=1e-3)
@click.option("--weight-decay", type=float, help="Weight decay", default=1e-5)
@click.option("--betas", type=float, nargs=2, help="Betas", default=(0.9, 0.95))
@click.option("--momentum", type=float, help="Momentum", default=0.9)
@click.option("--eps", type=float, help="Eps", default=1e-8)
@click.option("--total-epochs", type=int, help="Total epochs", default=10)
@click.option("--warmup-epochs", type=int, help="Warmup epochs", default=1)
@click.option("--log-times-per-epoch", type=int, help="Log times per epoch", default=10)
@click.option(
    "--ckpt-times-per-epoch", type=int, help="Checkpoint times per epoch", default=10
)
@click.option("--device", type=str, help="Device", default="cuda")
@click.option("--verbose", is_flag=True, help="Verbose")
@click.option("--image-size", type=int, help="Image size", default=224)
@click.option("--patch-size", type=int, help="Patch size", default=16)
@click.option(
    "--num-encoder-layers", type=int, help="Number of encoder layers", default=12
)
@click.option(
    "--num-decoder-layers", type=int, help="Number of decoder layers", default=12
)
@click.option("--num-heads", type=int, help="Number of heads", default=12)
@click.option("--hidden-size", type=int, help="Hidden size", default=768)
@click.option("--intermediate-size", type=int, help="Intermediate size", default=3072)
@click.option("--dropout", type=float, help="Dropout", default=0.1)
@click.option("--clear", is_flag=True, help="Clear logdir")
def train_model(**kwargs):
    if kwargs["clear"]:
        shutil.rmtree(kwargs["logdir"], ignore_errors=True)
    if kwargs["ddp"]:
        ddp.start(trainers.train, **kwargs)
    else:
        trainers.train(**kwargs)


if __name__ == "__main__":
    cli()
