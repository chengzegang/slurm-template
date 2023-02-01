__all__ = ["build"]

from . import vit


def build(**kwargs):
    if kwargs["model"] == "vit":
        return vit.build_vit(**kwargs)
    else:
        raise NotImplementedError
