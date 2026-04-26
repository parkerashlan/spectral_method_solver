import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from solver.fft2d import fft2d

sns.set_theme()
sns.set_palette("colorblind")
sns.set(font_scale=1.6)
plt.rcParams.update(
    {
        "font.size": 24,
        "axes.edgecolor": "0.15",
        "figure.figsize": (23, 23),
        "lines.marker": "o",
        "lines.linestyle": "",
        "lines.markersize": 2,
    }
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify energy conservation in the simulation."
    )
    parser.add_argument("--results", type=str, help="Path to the results file.")
    parser.add_argument("--params", type=str, help="Path to the parameters file.")
    return parser.parse_args()


def verify_energy(results, params):
    with open(params) as f:
        params = json.load(f)

    results = np.load(results)

    kinetic_energy = results["ke"]
    magnetic_energy = results["me"]
    total_energy = kinetic_energy + magnetic_energy

    vx = results["vx"]
    vy = results["vy"]
    bx = results["bx"]
    by = results["by"]
    k2 = results["k2"]
    nu = params["nu"]
    eta = params["eta"]
    Nx = params["Nx"]
    Ny = params["Ny"]
    k2 = k2[..., np.newaxis]

    # v * laplacian(v)
    vxhat = fft2d(vx, Nx, Ny, inverse=False)
    vyhat = fft2d(vy, Nx, Ny, inverse=False)
    # Compute laplacian in Fourier space
    lap_vxhat = k2 * vxhat
    lap_vyhat = k2 * vyhat
    # Convert back to physical space
    lap_vx = fft2d(lap_vxhat, Nx, Ny, inverse=True)
    lap_vy = fft2d(lap_vyhat, Nx, Ny, inverse=True)
    dkedt = -nu * (vx * lap_vx + vy * lap_vy)

    # b * laplacian(b)
    bxhat = fft2d(bx, Nx, Ny, inverse=False)
    byhat = fft2d(by, Nx, Ny, inverse=False)
    # Use k**2 since it's a Laplacian in x and y on bx and by
    lap_bxhat = k2 * bxhat
    lap_byhat = k2 * byhat
    lap_bx = fft2d(lap_bxhat, Nx, Ny, inverse=True)
    lap_by = fft2d(lap_byhat, Nx, Ny, inverse=True)
    dmedt = -eta * (bx * lap_bx + by * lap_by)

    dEdt = dkedt + dmedt
    int_dkedt = np.sum(dkedt, axis=(0, 1))
    int_dmedt = np.sum(dmedt, axis=(0, 1))
    int_dEdt = np.sum(dEdt, axis=(0, 1))

    ke_err = np.abs(kinetic_energy + int_dkedt)
    me_err = np.abs(magnetic_energy + int_dmedt)
    ed_err = np.abs(total_energy + int_dEdt)

    result = {
        "ke_err": ke_err,
        "me_err": me_err,
        "ed_err": ed_err,
        "ke": kinetic_energy,
        "me": magnetic_energy,
        "ed": total_energy,
        "int_dkedt": int_dkedt,
        "int_dmedt": int_dmedt,
        "int_dEdt": int_dEdt,
    }

    return result


def show_energy_plots(energy_result, save_path="./"):
    fig, ax = plt.subplots(3, 3)
    ax[0, 0].plot(energy_result["ke"][:-1], label="Kinetic Energy")
    ax[1, 0].plot(energy_result["me"][:-1], label="Magnetic Energy")
    ax[2, 0].plot(energy_result["ed"][:-1], label="Total Energy")
    ax[0, 0].legend()
    ax[1, 0].legend()
    ax[2, 0].legend()

    ax[0, 1].plot(energy_result["ke_err"][:-1], label="Kinetic Energy Error")
    ax[1, 1].plot(energy_result["me_err"][:-1], label="Magnetic Energy Error")
    ax[2, 1].plot(energy_result["ed_err"][:-1], label="Total Energy Error")
    ax[0, 1].legend()
    ax[1, 1].legend()
    ax[2, 1].legend()

    ax[0, 2].plot(energy_result["int_dkedt"][:-1], label="Kinetic Energy (Integrated)")
    ax[1, 2].plot(energy_result["int_dmedt"][:-1], label="Magnetic Energy (Integrated)")
    ax[2, 2].plot(energy_result["int_dEdt"][:-1], label="Total Energy (Integrated)")
    ax[0, 2].legend()
    ax[1, 2].legend()
    ax[2, 2].legend()

    plt.savefig(save_path + "energy_plots.png")


if __name__ == "__main__":
    args = parse_args()
    result = verify_energy(args.results, args.params)
    show_energy_plots(result, save_path="./")
