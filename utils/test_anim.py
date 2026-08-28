import argparse

import matplotlib.pyplot as plt
import numpy as np
from animate_data import AnimatePhysics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the solver results file")
    parser.add_argument("--output_path", type=str, help="Path to save the animation")
    parser.add_argument("--vorticity", action="store_true", help="Plot vorticity")
    parser.add_argument("--magnetic", action="store_true", help="Plot magnetic field")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data = np.load(args.data_path)

    animator = AnimatePhysics(
        data,
        interval=50,
        nt_save=1,
        x=data["x"],
        y=data["y"],
        rows=3,
        colorbar=True,
        frame_step=20,
        filename=args.output_path,
    )
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
        animator.plot_anim(plots)

    if args.vorticity:
        plots_vorticity = [
            {"data_key": "vorticity", "title": "Vorticity", "type": "field"},
            {
                "data_key": ("vx", "vy"),
                "title": "Velocity Streamplot",
                "type": "streamplot",
            },
        ]
        animator.plot_anim(plots_vorticity)


if __name__ == "__main__":
    main()
