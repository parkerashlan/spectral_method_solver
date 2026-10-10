import argparse
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

from utils.animate_data import AnimatePhysics

# def parse_args():
#    parser = argparse.ArgumentParser()
#    parser.add_argument("--data_path", type=str, help="Path to the solver results file")
#    parser.add_argument("--save_path", type=str, default="./streamplot_vs_psi.mp4")
#    parser.add_argument(
#        "--magnetic",
#        action="store_true",
#        help="Include magnetic field streamplot and potential A",
#    )
#    args = parser.parse_args()
#    return args
#
#
# def draw_frame(t, ax, data, x, y, norms, magnetic=False):
#    vx, vy, psi = data["vx"], data["vy"], data["psi"]
#
#    ax[0, 0].cla()
#    speed = np.sqrt(vx[:, :, t] ** 2 + vy[:, :, t] ** 2)
#    ax[0, 0].streamplot(
#        x,
#        y,
#        vx[:, :, t],
#        vy[:, :, t],
#        color=speed,
#        cmap="viridis",
#        norm=norms["stream"],
#        density=1.5,
#        linewidth=1.0,
#        broken_streamlines=False,
#    )
#    ax[0, 0].set_aspect("equal")
#    ax[0, 0].set_title(f"Velocity Streamplot (t={t})")
#
#    ax[0, 1].cla()
#    ax[0, 1].contourf(
#        x,
#        y,
#        psi[:, :, t],
#        levels=20,
#        cmap="viridis",
#        norm=norms["psi"],
#    )
#    ax[0, 1].set_aspect("equal")
#    ax[0, 1].set_title(f"Stream Function ψ (t={t})")
#
#    if magnetic:
#        bx, by, A = data["bx"], data["by"], data["A"]
#
#        ax[1, 0].cla()
#        b_speed = np.sqrt(bx[:, :, t] ** 2 + by[:, :, t] ** 2)
#        ax[1, 0].streamplot(
#            x,
#            y,
#            bx[:, :, t],
#            by[:, :, t],
#            color=b_speed,
#            cmap="viridis",
#            norm=norms["mag_stream"],
#            density=1.5,
#            linewidth=1.0,
#            broken_streamlines=False,
#        )
#        ax[1, 0].set_aspect("equal")
#        ax[1, 0].set_title(f"Magnetic Field Streamplot (t={t})")
#
#        ax[1, 1].cla()
#        ax[1, 1].contour(
#            x,
#            y,
#            A[:, :, t],
#            levels=20,
#            cmap="viridis",
#            norm=norms["A"],
#        )
#        ax[1, 1].set_aspect("equal")
#        ax[1, 1].set_title(f"Magnetic Potential A (t={t})")
#
#    return []


# def main():
#     args = parse_args()
#     data = np.load(args.data_path)
#     print(data.keys())
#
#     x, y = data["x"], data["y"]
#     vx, vy, psi = data["vx"], data["vy"], data["psi"]
#     Nt = psi.shape[-1]
#     magnetic = args.magnetic
#
#     speed_all = np.sqrt(vx**2 + vy**2)
#     norms = {
#         "stream": Normalize(vmin=np.nanmin(speed_all), vmax=np.nanmax(speed_all)),
#         "psi": Normalize(vmin=np.nanmin(psi), vmax=np.nanmax(psi)),
#     }
#
#     if magnetic:
#         bx, by, A = data["bx"], data["by"], data["A"]
#         b_speed_all = np.sqrt(bx**2 + by**2)
#         norms["mag_stream"] = Normalize(
#             vmin=np.nanmin(b_speed_all), vmax=np.nanmax(b_speed_all)
#         )
#         norms["A"] = Normalize(vmin=np.nanmin(A), vmax=np.nanmax(A))
#
#     rows = 2 if magnetic else 1
#     fig, ax = plt.subplots(rows, 2, figsize=(15, 7 * rows))
#     if rows == 1:
#         ax = ax[np.newaxis, :]
#
#     draw_frame(0, ax, data, x, y, norms, magnetic=magnetic)
#
#     colorbar_pairs = [("stream", ax[0, 0]), ("psi", ax[0, 1])]
#     if magnetic:
#         colorbar_pairs += [("mag_stream", ax[1, 0]), ("A", ax[1, 1])]
#
#     for key, a in colorbar_pairs:
#         sm = plt.cm.ScalarMappable(cmap="viridis", norm=norms[key])
#         sm.set_array([])
#         fig.colorbar(sm, ax=a)
#
#     anim = FuncAnimation(
#         fig,
#         partial(draw_frame, ax=ax, data=data, x=x, y=y, norms=norms, magnetic=magnetic),
#         frames=Nt,
#         interval=50,
#         blit=False,
#     )
#
#     if args.save_path.endswith(".gif"):
#         anim.save(args.save_path, writer="pillow")
#     else:
#         anim.save(args.save_path, writer="ffmpeg")
#     print(f"Animation saved to {args.save_path}")
#
#
# if __name__ == "__main__":
#     main()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the solver results file")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data = np.load(args.data_path)
    print(data.keys())
    psi = data["psi"]
    vorticity = data["vorticity"]
    vx = data["vx"]
    vy = data["vy"]
    # fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    # ax[0].streamplot(data["x"], data["y"], vx, vy, broken_streamlines=False)
    # ax[1].contourf(data["x"], data["y"], psi)
    bx = data["bx"]
    by = data["by"]
    kin_energy = data["ke"]
    mag_energy = np.sum((1 / 2) * (bx**2 + by**2), axis=(0, 1))
    fig, ax = plt.subplots(3, 1, figsize=(15, 10))
    tot_energy = data["ke"] + data["me"]
    animator = AnimatePhysics(
        data, interval=50, nt_save=10, x=data["x"], y=data["y"], rows=2, colorbar=True
    )
    plots = [
        {"data_key": "w", "title": "Vorticity", "type": "field"},
        # {"data_key": "bx", "title": "Magnetic Field Bx", "type": "field"},
    ]
    animator.plot_anim(plots)


# ax[0].plot(data["ke"][:-1] / np.max(data["ke"]), marker="o", markersize=1)
# ax[0].set_title("Kinetic Energy")
# ax[1].plot(mag_energy[:-1] / np.max(mag_energy), marker="o", markersize=1)
# ax[1].set_title("Magnetic Energy")
# ax[2].plot(tot_energy[:-1] / np.max(tot_energy), marker="o", markersize=1)
# ax[2].set_title("Total Energy")
# plt.show()
# show_streamplot(
#     bx,
#     by,
#     x=data["x"],
#     y=data["y"],
#     density=1.8,
#     linewidth=0.8,
#     save_path="./magnetic_field.mp4",
#     title="Magnetic Field t={t}",
#     nt_save=20,
#     pot_field=data["A"],
#     figsize=(12, 12),
# )


#     figsize=(15, 15),

if __name__ == "__main__":
    main()
