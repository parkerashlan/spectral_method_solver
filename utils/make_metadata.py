import datetime
import json
import types
from pathlib import Path

import numpy as np


def make_metadata(params, filename, dir="./", comment="", **kwargs):
    metadata = {}
    for varname, value in params:
        if not isinstance(value, (np.ndarray, types.FunctionType, types.LambdaType)):
            if isinstance(value, np.floating):
                metadata[varname] = float(value)
            else:
                metadata[varname] = value

    metadata["comment"] = comment

    for k, v in kwargs.items():
        metadata[k] = v

    with open(f"{dir}/{filename}", "w") as f:
        json.dump(metadata, f)

    return metadata


def make_run_dir(output_dir):
    cur_day = datetime.datetime.now().strftime("%Y-%m-%d")
    cur_time = datetime.datetime.now().strftime("%H-%M-%S")

    run_dir = Path(output_dir) / cur_day / cur_time
    Path(run_dir).mkdir(exist_ok=True, parents=True)

    viz_dir = run_dir / "viz"
    Path(viz_dir).mkdir(exist_ok=True)

    return run_dir, viz_dir
