from __future__ import annotations

import pytest
from torch.utils.data import Dataset

from project import datasets


@pytest.fixture(scope="module")
def datadir() -> str:
    return "tests/data"


@pytest.fixture(scope="module")
def image_folder(datadir: str) -> datasets.ImageFolder:
    return datasets.ImageFolder("tests/data", transform=None, target_transform=None)


def test_create_image_folder(image_folder: Dataset):
    assert hasattr(image_folder, "__len__")
    assert len(image_folder) == 15
