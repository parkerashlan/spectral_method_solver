import argparse
from pathlib import Path

import numpy as np

from utils.animate_data import AnimatePhysics
from utils.lazy_fields import open_fields


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Run directory or path to run.h5")
    parser.add_argument("--output_path", type=str, help="Path to save the animation")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--vorticity", action="store_true", help="Plot vorticity")
    mode_group.add_argument(
        "--magnetic", action="store_true", help="Plot magnetic field"
    )
    mode_group.add_argument(
        "--current_density", action="store_true", help="Plot current density"
    )
    mode_group.add_argument(
        "--plot_timesteps",
        action="store_true",
        help="Save individual plots for specific timesteps instead of animating",
    )
    parser.add_argument(
        "--timesteps",
        type=float,
        nargs="+",
        help="List of timesteps (in seconds) to save. Required with --plot_timesteps",
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        choices=["field", "streamplot", "contour"],
        help="Type of plot to save. Required with --plot_timesteps",
    )
    parser.add_argument(
        "--data_key",
        type=str,
        nargs="+",
        help=(
            "Data key to plot. Required with --plot_timesteps. "
            "Provide two keys (e.g. vx vy) for a streamplot."
        ),
    )

    parser.add_argument(
        "--filename",
        type=str,
        help="Output filename for the saved plot.",
    )
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        metavar=("y0", "y1", "x0", "x1"),
        help="Crop the saved plot to these bounds: y0 y1 x0 x1",
    )
    parser.add_argument(
        "--range_sample",
        type=int,
        default=None,
        help=(
            "Scan only this many snapshots when finding the colour range. "
            "Use ~10 for a quick look at a long high-resolution run."
        ),
    )
    parser.add_argument(
        "--colorbar_shrink",
        type=float,
        default=1.0,
        help=(
            "Fraction (0-1) to shrink the colorbar's length along the axes. "
            "Useful for wide/thin plots where the default colorbar is too big."
        ),
    )
    parser.add_argument(
        "--colorbar_fraction",
        type=float,
        default=0.15,
        help="Fraction of the axes size used for the colorbar's width.",
    )
    parser.add_argument(
        "--colorbar_aspect",
        type=float,
        default=20.0,
        help="Ratio of the colorbar's long side to its short side.",
    )
    parser.add_argument(
        "--colorbar_pad",
        type=float,
        default=0.04,
        help="Gap between the axes and the colorbar, as a fraction of the axes width.",
    )
    parser.add_argument(
        "--current_density_sq",
        action="store_true",
        help="Plot the current density squared (J^2) instead of the current density magnitude.",
    )
    args = parser.parse_args()

    if args.plot_timesteps and (
        args.timesteps is None or args.plot_type is None or args.data_key is None
    ):
        parser.error(
            "--plot_timesteps requires --timesteps, --plot_type, and --data_key"
        )

    return args


def resolve_run_file(data_path):
    """Accept either a run directory or the run.h5 file itself."""
    p = Path(data_path)
    if p.is_dir():
        p = p / "run.h5"
    if not p.exists():
        raise FileNotFoundError(f"no run file at {p}")
    return p


def main():
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    run_file = resolve_run_file(args.data_path)

    with open_fields(run_file) as data:
        attrs = data.attrs

        if not attrs.get("completed", True):
            print(f"WARNING: run did not complete: {attrs.get('error', 'unknown')}")
        print(
            f"{run_file}: {data.n_snapshots} snapshots, "
            f"{attrs['Ny']}x{attrs['Nx']}, "
            f"t = {data.times[0]:.4g} to {data.times[-1]:.4g}"
        )

        # Fields computed per frame from the stored A and psi.
        data.add_derived("j2", "jz", lambda a: a**2)

        plots = []
        if args.magnetic:
            plots = [
                {"data_key": "by", "title": "Magnetic Field By", "type": "field"},
                {"data_key": "bx", "title": "Magnetic Field Bx", "type": "field"},
                {
                    "data_key": ("bx", "by"),
                    "title": "Magnetic Field Streamplot",
                    "type": "streamplot",
                },
            ]

        if args.vorticity:
            plots = [
                {"data_key": "vorticity", "title": "Vorticity", "type": "field"},
                {
                    "data_key": ("vx", "vy"),
                    "title": "Velocity Streamplot",
                    "type": "streamplot",
                },
            ]

        if args.current_density:
            plots = [
                {"data_key": "jz", "title": "Current Density", "type": "field"},
                {
                    "data_key": ("vx", "vy"),
                    "title": "Velocity Streamplot",
                    "type": "streamplot",
                },
            ]

        if args.current_density_sq:
            plots = [
                {
                    "data_key": "j2",
                    "title": "Current Density Power $(J^2)$",
                    "type": "field",
                },
                {
                    "data_key": ("vx", "vy"),
                    "title": "Velocity Streamplot",
                    "type": "streamplot",
                },
            ]

        animator = AnimatePhysics(
            data,
            interval=100,
            x=data.x,
            y=data.y,
            times=data.times,
            rows=3,
            cols=1,
            panel_size=9.0,
            # vmin=-12,
            # vmax=12,
            colorbar=True,
            frame_step=10,
            save_path=str(output_path),
            filename=args.filename if args.filename else "animation.mp4",
            cmap="viridis",
            range_sample=args.range_sample,
            colorbar_kwargs={
                "shrink": args.colorbar_shrink,
                "fraction": args.colorbar_fraction,
                "aspect": args.colorbar_aspect,
                "pad": args.colorbar_pad,
            },
        )

        if args.plot_timesteps:
            data_key = (
                tuple(args.data_key) if len(args.data_key) > 1 else args.data_key[0]
            )
            animator.save_timesteps(
                args.timesteps,
                data_key,
                args.plot_type,
                save_path=str(output_path),
                bounds=args.bounds,
            )
        else:
            animator.plot_anim(plots, save_path=str(output_path))


if __name__ == "__main__":
    main()
