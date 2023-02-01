from __future__ import annotations

import os
from pathlib import Path


def get_version_dir(logdir: str, mkdirs: bool = True) -> str:
    if not os.path.exists(logdir) and not mkdirs:
        raise ValueError(f"{logdir} does not exist")
    os.makedirs(logdir, exist_ok=True)
    Path(logdir).mkdir(parents=True, exist_ok=True)
    existing_versions = [
        version.split("_")[1]
        for version in os.listdir(logdir)
        if version.startswith("version_")
    ]
    my_version = (
        0
        if len(existing_versions) == 0
        else max([int(version) for version in existing_versions]) + 1
    )
    version_dir = os.path.join(logdir, "version_" + str(my_version))
    Path(version_dir).mkdir(parents=True, exist_ok=True)
    return version_dir
