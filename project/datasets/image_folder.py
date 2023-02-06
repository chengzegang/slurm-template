from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import pandas as pd
import torch
from kornia.io import ImageLoadType, load_image
from torch.utils.data import Dataset

__RUST_EXT_NAME = "rust_ext"


def listdir(root: str, regex: str, recursive: bool = True) -> List[str]:

    if importlib.util.find_spec(__RUST_EXT_NAME) is not None:
        # If you choose to perform the actual import ...
        import rust_ext  # type: ignore

        print(f"{__RUST_EXT_NAME!r} has been imported")
        res: List[str] = rust_ext.listdir(root, regex, recursive)  # type: ignore
        return res
    else:
        print(f"{__RUST_EXT_NAME!r} not found, use python native implementation.")

    files: List[str] = []
    for root_, dirs, files in os.walk(root):
        for file in files:
            if re.match(regex, file):
                files.append(os.path.join(root_, file))
        if not recursive:
            break
    return files


class ImageFolder(Dataset):
    def __init__(
        self,
        root: str,
        regex: str = r"^.*\.(jpg|JPG|jpeg|JPEG|png|PNG)$",
        transforms: Callable[[torch.Tensor], torch.Tensor] | None = None,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.root = root
        self.transforms = transforms
        if verbose:
            print(f'scaning files under {root} with regex r"{regex}"...')
        paths = listdir(root, regex, recursive=True)
        if verbose:
            print(f"found {len(paths)} files.")
        self._paths = pd.DataFrame(paths, columns=["path"])

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path: str = self._paths["path"][idx]
        img: torch.Tensor = load_image(path, ImageLoadType.UNCHANGED, device="cuda")
        if self.transforms is not None:
            img = self.transforms(img)
        if img.dim() == 4:
            img = img.squeeze(0)
        return img
