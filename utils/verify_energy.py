import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.integrate import cumulative_trapezoid

from solver.fft2d import fft2d

mpl.rcParams["axes.formatter.limits"] = (-3, 4)
mpl.rcParams["axes.formatter.use_mathtext"] = True
mpl.rcParams["axes.formatter.useoffset"] = False
plt.close("all")
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
        "lines.markersize": 6,
    }
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify energy conservation in the simulation."
    )
    parser.add_argument("--results", type=str, help="Path to the results file.")
    parser.add_argument("--params", type=str, help="Path to the parameters file.")
    parser.add_argument("--save_path", type=str, help="Path to save plot.")
    parser.add_argument(
        "--float32",
        action="store_true",
        help=(
            "Use single-precision (float32) for the energy calculations to reduce "
            "memory usage on long runs. Defaults to double precision (float64), "
            "which is recommended for resolving small conservation residuals."
        ),
    )
    parser.add_argument(
        "--mmap",
        action="store_true",
        help=(
            "Use verify_energy_mmap instead of verify_energy. Memory-maps the "
            "results file and processes one timestep at a time, which avoids "
            "holding many full (Nx, Ny, Nt)-sized arrays in RAM at once. Use this "
            "for large/long runs that OOM with the regular vectorized version."
        ),
    )
    return parser.parse_args()


def verify_energy(results, params, save_path, use_float32=False):
    real_dtype = np.float32 if use_float32 else np.float64
    complex_dtype = np.complex64 if use_float32 else np.complex128

    with open(params) as f:
        params = json.load(f)

    results = np.load(results)

    # magnetic_energy = results["me"]
    # total_energy = kinetic_energy + magnetic_energy

    vx = results["vx"].astype(real_dtype)
    vy = results["vy"].astype(real_dtype)
    bx = results["bx"].astype(real_dtype)
    by = results["by"].astype(real_dtype)
    k2 = results["k2"].astype(real_dtype)
    nu = params["nu"]
    eta = params["eta"]
    Nx = params["Nx"]
    Ny = params["Ny"]
    dt = params["dt"]
    Lx = params["Lx"]
    Ly = params["Ly"]

    k2 = k2[..., np.newaxis]
    kx = results["kx"].astype(real_dtype)
    ky = results["ky"].astype(real_dtype)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    kx = kx_grid[..., np.newaxis]
    ky = ky_grid[..., np.newaxis]
    dx = Lx / Nx
    dy = Ly / Ny

    # kinetic_energy = results["ke"] * dx * dy
    # magnetic_energy = results["me"] * dx * dy
    # kinetic_energy = np.sum(results["ke"], axis=(0, 1)) * dx * dy
    # magnetic_energy = np.sum(results["me"], axis=(0, 1)) * dx * dy
    kinetic_energy = (1 / 2) * np.sum(vx**2 + vy**2, axis=(0, 1)) * dx * dy
    magnetic_energy = (1 / 2) * np.sum(bx**2 + by**2, axis=(0, 1)) * dx * dy
    total_energy = kinetic_energy + magnetic_energy

    # v * laplacian(v)
    vxhat = fft2d(vx, Nx, Ny, inverse=False)
    vyhat = fft2d(vy, Nx, Ny, inverse=False)
    # Compute laplacian in Fourier space
    # k2_mult = k2.copy()

    k2[0, 0] = 0
    lap_vxhat = -k2 * vxhat
    lap_vyhat = -k2 * vyhat
    # Convert back to physical space
    lap_vx = fft2d(lap_vxhat, Nx, Ny, inverse=True)
    lap_vy = fft2d(lap_vyhat, Nx, Ny, inverse=True)

    # Viscous Dissipation term
    dkedt = (vx * lap_vx + vy * lap_vy) * nu
    del vxhat, vyhat, lap_vxhat, lap_vyhat
    # b * laplacian(b)
    bxhat = fft2d(bx, Nx, Ny, inverse=False)
    byhat = fft2d(by, Nx, Ny, inverse=False)

    # v * (j x B)
    i_complex = complex_dtype(1j)
    dbxdy_hat = i_complex * ky * bxhat
    dbydx_hat = i_complex * kx * byhat
    dbxdy = fft2d(dbxdy_hat, Nx, Ny, inverse=True)
    dbydx = fft2d(dbydx_hat, Nx, Ny, inverse=True)
    curlB = dbydx - dbxdy
    # j x B = (j_z z_hat) x B = (-j_z*By, j_z*Bx), so
    # v . (j x B) = j_z * (vy*Bx - vx*By)
    vdotjcrossB = vy * bx * curlB - vx * by * curlB
    del dbxdy_hat, dbydx_hat, dbxdy, dbydx, vx, vy

    dkedt = dkedt + vdotjcrossB

    # Use k**2 since it's a Laplacian in x and y on bx and by
    lap_bxhat = -k2 * bxhat
    lap_byhat = -k2 * byhat
    lap_bx = fft2d(lap_bxhat, Nx, Ny, inverse=True)
    lap_by = fft2d(lap_byhat, Nx, Ny, inverse=True)
    dbedt = eta * (bx * lap_bx + by * lap_by) - vdotjcrossB
    del lap_bxhat, lap_byhat, lap_bx, lap_by, bx, by, kx, ky, k2

    dEdt = dkedt + dbedt
    intdxdy_dEdt = np.sum(dEdt, axis=(0, 1)) * dx * dy
    intdxdy_dkedt = np.sum(dkedt, axis=(0, 1)) * dx * dy
    intdxdy_dbedt = np.sum(dbedt, axis=(0, 1)) * dx * dy

    trap_dkedt = cumulative_trapezoid(
        intdxdy_dkedt, dx=dt
    )  # integrates kinetic energy up to that timestep
    ke0 = kinetic_energy[0]
    intt_dkedt = ke0 + trap_dkedt

    trap_dbedt = cumulative_trapezoid(
        intdxdy_dbedt, dx=dt
    )  # integrates magnetic energy up to that timestep
    me0 = magnetic_energy[0]
    intt_dbedt = me0 + trap_dbedt

    trap_dEdt = cumulative_trapezoid(
        intdxdy_dEdt, dx=dt
    )  # integrates total energy up to that timestep
    ed0 = total_energy[0]
    intt_dEdt = ed0 + trap_dEdt

    num_dkedt = np.gradient(kinetic_energy, dt, edge_order=2)
    num_dbedt = np.gradient(magnetic_energy, dt, edge_order=2)

    # cumulative_trapezoid (no `initial`) returns length N-1, where element i is the
    # integral from t0 to t_{i+1}. Adding E[0] yields the predicted energy at
    # t_{i+1}, so compare against E[1:] (not E[:-1]) to stay time-aligned.
    ke_err = np.abs(kinetic_energy[1:] - intt_dkedt)
    me_err = np.abs(magnetic_energy[1:] - intt_dbedt)
    ed_err = np.abs(total_energy[1:] - intt_dEdt)

    result = {
        "ke_err": ke_err,
        "me_err": me_err,
        "ed_err": ed_err,
        "ke": kinetic_energy,
        "me": magnetic_energy,
        "ed": total_energy,
        "int_dkedt": intt_dkedt,
        "int_dmedt": intt_dbedt,
        "num_dkedt": num_dkedt,
        "num_dbedt": num_dbedt,
        "dkedt": intdxdy_dkedt,
        "dbedt": intdxdy_dbedt,
        "trap_dkedt": trap_dkedt,
        "int_dEdt": intt_dEdt,
        "dt": dt,
    }

    return result


def _frame_energy_terms(vx, vy, bx, by, kx, ky, k2, nu, eta, Nx, Ny, dx, dy, i_complex):
    """Compute per-frame (single-timestep) energy quantities.

    Operates on one 2D (Nx, Ny) slice at a time so that peak memory stays at
    O(Nx * Ny) instead of O(Nx * Ny * Nt), unlike the fully vectorized
    `verify_energy`, which builds many full (Nx, Ny, Nt)-sized intermediates.
    """
    vxhat = fft2d(vx, Nx, Ny, inverse=False)
    vyhat = fft2d(vy, Nx, Ny, inverse=False)
    lap_vx = fft2d(-k2 * vxhat, Nx, Ny, inverse=True)
    lap_vy = fft2d(-k2 * vyhat, Nx, Ny, inverse=True)

    # Viscous dissipation term
    dkedt = (vx * lap_vx + vy * lap_vy) * nu

    bxhat = fft2d(bx, Nx, Ny, inverse=False)
    byhat = fft2d(by, Nx, Ny, inverse=False)

    dbxdy = fft2d(i_complex * ky * bxhat, Nx, Ny, inverse=True)
    dbydx = fft2d(i_complex * kx * byhat, Nx, Ny, inverse=True)
    curlB = dbydx - dbxdy
    # j x B = (j_z z_hat) x B = (-j_z*By, j_z*Bx), so
    # v . (j x B) = j_z * (vy*Bx - vx*By)
    vdotjcrossB = vy * bx * curlB - vx * by * curlB

    dkedt = dkedt + vdotjcrossB

    lap_bx = fft2d(-k2 * bxhat, Nx, Ny, inverse=True)
    lap_by = fft2d(-k2 * byhat, Nx, Ny, inverse=True)
    dbedt = eta * (bx * lap_bx + by * lap_by) - vdotjcrossB

    ke = 0.5 * np.sum(vx**2 + vy**2) * dx * dy
    me = 0.5 * np.sum(bx**2 + by**2) * dx * dy
    dkedt_int = np.sum(dkedt) * dx * dy
    dbedt_int = np.sum(dbedt) * dx * dy

    return ke, me, dkedt_int, dbedt_int


def verify_energy_mmap(results, params, save_path, use_float32=False):
    """Memory-mapped, frame-by-frame version of `verify_energy`.

    Identical physics/math to `verify_energy`, but the results `.npz` is
    memory-mapped and only one 2D (Nx, Ny) timestep is materialized at a
    time, instead of vectorizing over the full (Nx, Ny, Nt) arrays. This
    keeps peak memory roughly constant regardless of the number of saved
    timesteps, which matters for large/long runs (e.g. many GB npz files)
    that OOM with the vectorized `verify_energy`.
    """
    real_dtype = np.float32 if use_float32 else np.float64
    complex_dtype = np.complex64 if use_float32 else np.complex128
    i_complex = complex_dtype(1j)

    with open(params) as f:
        params = json.load(f)

    # mmap_mode requires the npz to be uncompressed (np.savez, not
    # np.savez_compressed); each array is then read lazily from disk.
    results = np.load(results, mmap_mode="r")

    vx_all = results["vx"]
    vy_all = results["vy"]
    bx_all = results["bx"]
    by_all = results["by"]

    nu = params["nu"]
    eta = params["eta"]
    Nx = params["Nx"]
    Ny = params["Ny"]
    dt = params["dt"]
    Lx = params["Lx"]
    Ly = params["Ly"]
    dx = Lx / Nx
    dy = Ly / Ny

    Nt = vx_all.shape[-1]

    k2 = results["k2"].astype(real_dtype).copy()
    k2[0, 0] = 0

    kx_1d = results["kx"].astype(real_dtype)
    ky_1d = results["ky"].astype(real_dtype)
    kx, ky = np.meshgrid(kx_1d, ky_1d)

    kinetic_energy = np.empty(Nt, dtype=np.float64)
    magnetic_energy = np.empty(Nt, dtype=np.float64)
    intdxdy_dkedt = np.empty(Nt, dtype=np.float64)
    intdxdy_dbedt = np.empty(Nt, dtype=np.float64)

    for t in range(Nt):
        vx = np.array(vx_all[:, :, t], dtype=real_dtype)
        vy = np.array(vy_all[:, :, t], dtype=real_dtype)
        bx = np.array(bx_all[:, :, t], dtype=real_dtype)
        by = np.array(by_all[:, :, t], dtype=real_dtype)

        ke, me, dkedt_int, dbedt_int = _frame_energy_terms(
            vx, vy, bx, by, kx, ky, k2, nu, eta, Nx, Ny, dx, dy, i_complex
        )
        kinetic_energy[t] = ke
        magnetic_energy[t] = me
        intdxdy_dkedt[t] = dkedt_int
        intdxdy_dbedt[t] = dbedt_int

    total_energy = kinetic_energy + magnetic_energy
    intdxdy_dEdt = intdxdy_dkedt + intdxdy_dbedt

    trap_dkedt = cumulative_trapezoid(
        intdxdy_dkedt, dx=dt
    )  # integrates kinetic energy up to that timestep
    ke0 = kinetic_energy[0]
    intt_dkedt = ke0 + trap_dkedt

    trap_dbedt = cumulative_trapezoid(
        intdxdy_dbedt, dx=dt
    )  # integrates magnetic energy up to that timestep
    me0 = magnetic_energy[0]
    intt_dbedt = me0 + trap_dbedt

    trap_dEdt = cumulative_trapezoid(
        intdxdy_dEdt, dx=dt
    )  # integrates total energy up to that timestep
    ed0 = total_energy[0]
    intt_dEdt = ed0 + trap_dEdt

    num_dkedt = np.gradient(kinetic_energy, dt, edge_order=2)
    num_dbedt = np.gradient(magnetic_energy, dt, edge_order=2)

    # cumulative_trapezoid (no `initial`) returns length N-1, where element i is the
    # integral from t0 to t_{i+1}. Adding E[0] yields the predicted energy at
    # t_{i+1}, so compare against E[1:] (not E[:-1]) to stay time-aligned.
    ke_err = np.abs(kinetic_energy[1:] - intt_dkedt)
    me_err = np.abs(magnetic_energy[1:] - intt_dbedt)
    ed_err = np.abs(total_energy[1:] - intt_dEdt)

    result = {
        "ke_err": ke_err,
        "me_err": me_err,
        "ed_err": ed_err,
        "ke": kinetic_energy,
        "me": magnetic_energy,
        "ed": total_energy,
        "int_dkedt": intt_dkedt,
        "int_dmedt": intt_dbedt,
        "num_dkedt": num_dkedt,
        "num_dbedt": num_dbedt,
        "dkedt": intdxdy_dkedt,
        "dbedt": intdxdy_dbedt,
        "trap_dkedt": trap_dkedt,
        "int_dEdt": intt_dEdt,
        "dt": dt,
    }

    return result


def show_energy_plots(energy_result, save_path: str | Path = "."):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    plt.close("all")
    fig, ax = plt.subplots(5, 2, figsize=(15, 22))
    t = np.arange(len(energy_result["ke"])) * energy_result["dt"]
    # Row 0: raw energy -- col 0 = Kinetic, col 1 = Magnetic
    ax[0, 0].plot(t[:-1], energy_result["ke"][:-1], label="Kinetic Energy", alpha=0.5)
    ax[0, 0].legend()

    ax[0, 1].plot(t[:-1], energy_result["me"][:-1], label="Magnetic Energy", alpha=0.5)
    ax[0, 1].legend()

    # Row 1: energy integrated in time -- col 0 = Kinetic, col 1 = Magnetic
    ax[1, 0].plot(
        t[:-1],
        energy_result["int_dkedt"],
        label="Kinetic Energy (Integrated in time)",
        alpha=0.5,
    )
    ax[1, 0].legend()

    ax[1, 1].plot(
        t[:-1],
        energy_result["int_dmedt"],
        label="Magnetic Energy (Integrated in time)",
        alpha=0.5,
    )
    ax[1, 1].legend()

    # Row 2: dE/dt (numerical derivative) -- col 0 = Kinetic, col 1 = Magnetic
    ax[2, 0].plot(
        t[:-2],
        energy_result["num_dkedt"][:-2],
        label="Kinetic Energy Derivative",
        alpha=0.5,
    )
    ax[2, 0].legend()

    ax[2, 1].plot(
        t[:-2],
        energy_result["num_dbedt"][:-2],
        label="Magnetic Energy Derivative",
        alpha=0.5,
    )
    ax[2, 1].legend()

    # Row 3: dE/dt integrated in space -- col 0 = Kinetic, col 1 = Magnetic
    ax[3, 0].plot(
        t[:-1],
        energy_result["dkedt"][:-1],
        label="Kinetic Energy (dK/dt integrated in space)",
        alpha=0.5,
    )
    ax[3, 0].legend()

    ax[3, 1].plot(
        t[:-1],
        energy_result["dbedt"][:-1],
        label="Magnetic Energy (dB/dt integrated in space)",
        alpha=0.5,
    )
    ax[3, 1].legend()

    # Row 4: error (log scale) -- col 0 = Kinetic, col 1 = Magnetic
    ax[4, 0].semilogy(
        t[:-2], energy_result["ke_err"][:-1], label="Kinetic Energy Error", alpha=0.5
    )
    ax[4, 0].semilogy(
        t[:-2],
        np.abs(energy_result["num_dkedt"] - energy_result["dkedt"])[:-2],
        label="Kinetic Energy Derivative Error",
        alpha=0.5,
    )
    ax[4, 0].legend()

    ax[4, 1].semilogy(
        t[:-2], energy_result["me_err"][:-1], label="Magnetic Energy Error", alpha=0.5
    )
    ax[4, 1].semilogy(
        t[:-2],
        np.abs(energy_result["num_dbedt"] - energy_result["dbedt"])[:-2],
        label="Magnetic Energy Derivative Error",
        alpha=0.5,
    )
    ax[4, 1].legend()

    fig2, ax2 = plt.subplots(2, 1, figsize=(15, 15))
    ke_deriv_error = np.abs(energy_result["num_dkedt"] - energy_result["dkedt"])[:-2]
    me_deriv_error = np.abs(energy_result["num_dbedt"] - energy_result["dbedt"])[:-2]
    ax2[0].plot(
        t[:-2], energy_result["ke_err"][:-1], label="Kinetic Energy Error", alpha=0.5
    )
    ax2[1].plot(
        t[:-2], energy_result["me_err"][:-1], label="Magnetic Energy Error", alpha=0.5
    )
    ax2[0].plot(
        t[:-2], ke_deriv_error, label="Kinetic Energy Derivative Error", alpha=0.5
    )
    ax2[1].plot(
        t[:-2], me_deriv_error, label="Magnetic Energy Derivative Error", alpha=0.5
    )
    ax2[0].legend()
    ax2[1].legend()
    ax2[0].set_title(f"Error with step size ({energy_result['dt']})")
    ax2[1].set_title(f"Error with step size ({energy_result['dt']})")

    plt.tight_layout()
    fig2.tight_layout()
    fig.savefig(save_path / "energy.png")
    fig2.savefig(save_path / "energy_error.png")
    plt.close(fig)
    plt.close(fig2)


if __name__ == "__main__":
    args = parse_args()
    parent_path = Path(args.results).parent
    save_path = parent_path / "viz"
    print(save_path)
    verify_energy_fn = verify_energy_mmap if args.mmap else verify_energy
    result = verify_energy_fn(
        args.results, args.params, save_path, use_float32=args.float32
    )
    show_energy_plots(result, save_path=save_path)
