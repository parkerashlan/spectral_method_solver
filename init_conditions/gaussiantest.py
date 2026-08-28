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

OUTPUT_DIR = "gaussiantest_outputs"
cur_run_dir, viz_dir = make_run_dir(OUTPUT_DIR)


def gaussian2d(x, y, mu, sigma):
    return (1 / (2 * np.pi * sigma**2)) * np.exp(
        -((x - mu) ** 2) / (2 * sigma**2) - (y - mu) ** 2 / (2 * sigma**2)
    )


mu = 0
sigma = 1

w0 = lambda x, y: -y * np.exp(-(x**2) - y**2)

j0 = lambda x, y: x * 0 + y * 0

params = Params(
    Nx=256,
    Ny=256,
    Lx=2.0 * np.pi,
    Ly=2.0 * np.pi,
    dt=(1.0 / 664.0) * 0.01,
    nt=1200,
    nt_save=1,
    w0=w0,
    j0=j0,
    eta=1e-2,
    nu=1e-2,
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
    # "current_sheet_strength": 1.0,
    # "wall_loc1": 0.5,
    # "wall_loc2": -0.5,
    # "sign1": 1,
    # "sign2": -1,
    "solve_time": end - start,
    "notes": "No current but add viscosity/resistivity, Gaussian initial vorticity",
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
