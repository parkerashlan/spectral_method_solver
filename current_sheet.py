import time

import numpy as np

# from data_analysis import mag_energy
from solver.fft_solver import FFT2DSolver
from utils.animate import (
    animate_quiver,
    animate_streamplot,
    show_animation,
    show_quiver,
    show_streamplot,
)
from utils.make_metadata import make_metadata, make_run_dir
from utils.params import Params


def current_sheet(x, y, strength, sheet_loc, sign):
    """
    Creates a current sheet centered at `sheet_loc`.

    Parameters
    ----------
    x : array-like
        The x-coordinates.
    y : array-like
        The y-coordinates.
    strength : float
        The strength of the current sheet.
    sheet_loc : float
        The location of the sheet.
    sign : int
        The sign of the current sheet (1 or -1).

    Returns
    -------
    array-like
        The current sheet values at (x, y).
    """
    if sign not in [1, -1]:
        raise ValueError("sign must be 1 or -1")
    return strength * (sign * (1 / np.cosh(strength * (x + sheet_loc)) ** 2))


OUTPUT_DIR = "current_sheets_output"
cur_run_dir, viz_dir = make_run_dir(OUTPUT_DIR)

u0 = 0.1
w0 = lambda x, y: -2 * np.pi * u0 * np.cos(2 * np.pi * y)

j0 = lambda x, y: (
    current_sheet(x, y, strength=1.0, sheet_loc=0.5, sign=1)
    + current_sheet(x, y, strength=1.0, sheet_loc=-0.5, sign=-1)
)


params = Params(
    Nx=256,
    Ny=256,
    Lx=2.0,
    Ly=2.0,
    dt=1.0 / 664.0,
    nt=2000,
    nt_save=10,
    w0=w0,
    j0=j0,
    eta=1e-5,
    nu=1e-5,
)
print(params.dt)

# fig, ax = plt.subplots()
# ax.plot(params.x, params.j[0, :])
# plt.show()

start = time.time()
solver = FFT2DSolver(params)
solver_results = solver.solve()
end = time.time()
print(f"Solve time: {end - start:.2f} seconds")

extra_metadata = {
    "current_sheet_strength": 10.0,
    "wall_loc1": 0.5,
    "wall_loc2": -0.5,
    "sign1": 1,
    "sign2": -1,
    "solve_time": end - start,
    "show_potential_field": True,
}
make_metadata(
    params,
    "params.json",
    dir=cur_run_dir,
    comment="Parameters for the current sheet simulation",
    **extra_metadata,
)

wzq = solver_results["vorticity"]
jzq = solver_results["jz"]
cfl = solver_results["cfl"]
vx = solver_results["vx"]
vy = solver_results["vy"]
kin_energy = np.sum((1 / 2) * (vx**2 + vy**2), axis=(0, 1))
psi = solver_results["psi"]
bx = solver_results["bx"]
by = solver_results["by"]
A = solver_results["A"]
mag_energy = np.sum((1 / 2) * (bx**2 + by**2), axis=(0, 1))

np.savez(
    cur_run_dir / "solver_results.npz",
    **solver_results,
    ke=kin_energy,
    me=mag_energy,
    x=params.x,
    y=params.y,
)

# show_animation(
#     wzq,
#     cmap="jet",
#     vmin=min_init,
#     vmax=max_init,
#     title="Vorticity t={t}",
#     colorbar=True,
#     figsize=(8, 6),
#     save_path=str(viz_dir / MP4_NAME),
#     nt_save=params.nt_save,
# )

# show_animation(
#     kin_energy,
#     cmap="jet",
#     vmin=np.min(kin_energy[:, :, 0]),
#     vmax=np.max(kin_energy[:, :, 0]),
#     title="Kinetic Energy t={t}",
#     colorbar=True,
#     figsize=(8, 6),
#     save_path=str(viz_dir / "kinetic_energy.mp4"),
#     nt_save=params.nt_save,
# )

# show_animation(
#     jzq,
#     x=params.x,
#     y=params.y,
#     cmap="jet",
#     vmin=np.min(jzq[:, :, 0]),
#     vmax=np.max(jzq[:, :, 0]),
#     title="Current Density t={t}",
#     colorbar=True,
#     figsize=(8, 6),
#     save_path=str(viz_dir / "current_density.mp4"),
#     nt_save=params.nt_save,
# )

# animate_streamplot(
#     vx,
#     vy,
#     x=params.x,
#     y=params.y,
#     title="Velocity t={t}",
#     save_path=str(viz_dir / "velocity_streamplot.mp4"),
#     nt_save=params.nt_save,
#     pot_field=psi,
# )
#
# animate_streamplot(
#     bx,
#     by,
#     x=params.x,
#     y=params.y,
#     title="Magnetic Field t={t}",
#     save_path=str(viz_dir / "magnetic_streamplot.mp4"),
#     nt_save=params.nt_save,
#     pot_field=A,
# )

show = False
if show:
    show_quiver(
        vx,
        vy,
        x=params.x,
        y=params.y,
        stride=15,
        title="Velocity t={t}",
        save_path=str(viz_dir / "velocity_quiver.mp4"),
    )
    show_quiver(
        bx,
        by,
        x=params.x,
        y=params.y,
        stride=15,
        title="Magnetic Field t={t}",
        save_path=str(viz_dir / "magnetic_quiver.mp4"),
    )

if show:
    show_streamplot(
        vx,
        vy,
        x=params.x,
        y=params.y,
        title="Velocity t={t}",
        save_path=str(viz_dir / "velocity_streamplot.mp4"),
    )
    show_streamplot(
        bx,
        by,
        x=params.x,
        y=params.y,
        title="Magnetic Field t={t}",
        save_path=str(viz_dir / "mag_field_streamplot.mp4"),
    )
