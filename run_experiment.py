import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np

from solver.fft_solver import FFT2DSolver
from utils.config import load_config
from utils.make_metadata import make_metadata, make_run_dir
from utils.params import Params


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str, help="Path to config file")
    group.add_argument(
        "--show-init-conds", action="store_true", help="Show initial conditions"
    )
    parser.add_argument(
        "--output-dir", type=str, required=False, help="Output directory"
    )
    return parser.parse_args()


def load_init_condition(init_condition_path):
    module_name = init_condition_path.stem
    spec = importlib.util.spec_from_file_location(module_name, init_condition_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(args):
    init_conds_path = Path("./init_conditions")
    init_conds = [
        f for f in init_conds_path.iterdir() if f.is_file()
    ]  # gather all initial condition file names

    if args.show_init_conds:
        for cond in init_conds:
            name = cond.name.split("/")[-1]
            print(name)
        return

    cfg = load_config(args.config)

    # Load function for initial condition
    init_condition = cfg["initial_conditions"]
    print(init_condition)
    try:
        init_condition_path = [
            p for p in init_conds if init_condition["name"] in p.name
        ][0]
    except IndexError:
        raise ValueError(f"Initial condition '{init_condition['name']}' not found")
    init_condition_module = load_init_condition(init_condition_path)
    w0, A0 = getattr(init_condition_module, "init_functions")()
    params = Params(w0=w0, A0=A0, **cfg["params"])

    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    else:
        OUTPUT_DIR = f"outputs/{cfg['initial_conditions']['name']}_outputs"

    Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)
    cur_run_dir, viz_dir = make_run_dir(OUTPUT_DIR)

    make_metadata(
        params,
        "params.json",
        dir=str(Path(cur_run_dir).resolve()),
        comment="Parameters for the current sheet simulation",
        **cfg.get("extra_metadata", {}),
    )

    start = time.time()
    solver = FFT2DSolver(params)
    solver_results = solver.solve(outfile=Path(cur_run_dir) / "run.h5")
    end = time.time()
    print(f"Solve time: {end - start:.2f} seconds")
    print(f"Saved results to {cur_run_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
