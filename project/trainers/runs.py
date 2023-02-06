from __future__ import annotations

import os
from pathlib import Path


def get_run_dir(logdir: str, mkdirs: bool = True) -> str:
    if not os.path.exists(logdir) and not mkdirs:
        raise ValueError(f"{logdir} does not exist")
    os.makedirs(logdir, exist_ok=True)
    Path(logdir).mkdir(parents=True, exist_ok=True)
    existing_runs = [
        run.split("_")[1] for run in os.listdir(logdir) if run.startswith("run_")
    ]
    my_run = (
        0 if len(existing_runs) == 0 else max([int(run) for run in existing_runs]) + 1
    )
    run_dir = os.path.join(logdir, "run_" + str(my_run))
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    return run_dir
