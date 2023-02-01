__all__ = ["build", "train"]

from . import denoise


def build(**kwargs):
    if kwargs["trainer"] == "denoise":
        return denoise.build(**kwargs)
    else:
        raise NotImplementedError


def train(**kwargs):
    trainer = build(**kwargs)
    trainer.train()
