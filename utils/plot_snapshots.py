"""
Plot jz^2 for saved snapshots of a run.

Usage (run from the project root so `solver` is importable):
    python -m utils.plot_snapshots path/to/run.h5            # only the save_steps snapshots
    python -m utils.plot_snapshots path/to/run.h5 --all      # every saved snapshot
    python -m utils.plot_snapshots path/to/run.h5 --bounds -2.5 2.5 -0.15 0.15
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from solver.io_hdf5 import RunReader


def snapshot_indices(r, only_save_steps=True):
    """Indices of snapshots to plot (those matching params.save_steps by default)."""
    if not only_save_steps or "save_steps" not in r.f["grid"]:
        return list(range(r.n_snapshots))

    dt = float(r.attrs["dt"])
    times = r.t[:]
    idx = []
    for step in r.f["grid/save_steps"][:]:
        i = int(np.argmin(np.abs(times - step * dt)))
        if abs(times[i] - step * dt) < 0.5 * dt:
            idx.append(i)
        else:
            print(f"Warning: step {step} not found in saved snapshots")
    return idx


def plot_jz2(r, i, bounds, out_dir, x, y, ix, iy):
    jz = r.current_density(i)
    jz2 = (jz**2)[np.ix_(iy, ix)]

    width = 12.0
    height = width * (bounds[3] - bounds[2]) / (bounds[1] - bounds[0])
    fig, ax = plt.subplots(figsize=(width, height + 2.0))
    im = ax.pcolormesh(x[ix], y[iy], jz2, shading="auto", cmap="jet")
    ax.set_aspect("equal")
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"$j_z^2$, t = {r.t[i]:.4f} (snapshot {i})")
    fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.3, aspect=40)

    step = int(round(r.t[i] / float(r.attrs["dt"])))
    fig.savefig(out_dir / f"jz2_step{step:06d}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_file", type=str)
    p.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        default=[-2.5, 2.5, -0.15, 0.15],
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
    )
    p.add_argument("--all", action="store_true", help="plot every saved snapshot")
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()

    run_file = Path(args.run_file)
    out_dir = Path(args.out_dir) if args.out_dir else run_file.parent / "viz"
    out_dir.mkdir(exist_ok=True, parents=True)

    with RunReader(run_file) as r:
        x = r.f["grid/x"][:]
        y = r.f["grid/y"][:]
        xmin, xmax, ymin, ymax = args.bounds
        ix = np.where((x >= xmin) & (x <= xmax))[0]
        iy = np.where((y >= ymin) & (y <= ymax))[0]

        for i in snapshot_indices(r, only_save_steps=not args.all):
            plot_jz2(r, i, args.bounds, out_dir, x, y, ix, iy)
            print(f"Saved plot for snapshot {i} (t = {r.t[i]:.4f})")


if __name__ == "__main__":
    main()
