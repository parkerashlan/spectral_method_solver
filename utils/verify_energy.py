import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas._libs.lib import i8max
from scipy.integrate import cumulative_trapezoid

from solver.fft2d import fft2d
from utils.lazy_fields import open_fields

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
    parser.add_argument("--data_path", type=str, help="Run directory or path to run.h5")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Directory to save plots. Defaults to <run directory>/viz.",
    )
    parser.add_argument(
        "--alfven",
        action="store_true",
        help=(
            "Also plot the velocity amplitude over time and vy line-outs along x, "
            "to check that an Alfven wave travelling in x keeps its amplitude."
        ),
    )
    return parser.parse_args()


def resolve_run_file(data_path):
    """Accept either a run directory or the run.h5 file itself."""
    p = Path(data_path)
    if p.is_dir():
        p = p / "run.h5"
    if not p.exists():
        raise FileNotFoundError(f"no run file at {p}")
    return p


def _frame_energy_terms(vx, vy, bx, by, kx, ky, k2, nu, eta, Nx, Ny, dx, dy):
    """Compute per-frame (single-snapshot) energy quantities.

    All field arrays are 2D with shape (Ny, Nx). kx, ky, and k2 are broadcastable
    to the rfft spectral shape (Ny, Nx // 2 + 1).
    """
    vxhat = fft2d(vx, Nx, Ny, inverse=False)
    vyhat = fft2d(vy, Nx, Ny, inverse=False)
    lap_vx = fft2d(-k2 * vxhat, Nx, Ny, inverse=True)
    lap_vy = fft2d(-k2 * vyhat, Nx, Ny, inverse=True)

    # Viscous dissipation term
    dkedt = (vx * lap_vx + vy * lap_vy) * nu

    bxhat = fft2d(bx, Nx, Ny, inverse=False)
    byhat = fft2d(by, Nx, Ny, inverse=False)

    dbxdy = fft2d(1j * ky * bxhat, Nx, Ny, inverse=True)
    dbydx = fft2d(1j * kx * byhat, Nx, Ny, inverse=True)
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


def mag_helicity(data):
    attrs = data.attrs
    Lx = attrs["Lx"]
    Ly = attrs["Ly"]
    Nx = int(attrs["Nx"])
    Ny = int(attrs["Ny"])
    dx = Lx / Nx
    dy = Ly / Ny
    Nt = data.n_snapshots

    mag_helicity = np.empty(Nt)
    for i in range(Nt):
        mag_pot = data["A"].frame(i)
        mag_helicity[i] = mag_pot.sum() * dx * dy

    return mag_helicity


def cross_helicity(data):
    attrs = data.attrs
    Lx = attrs["Lx"]
    Ly = attrs["Ly"]
    Nx = int(attrs["Nx"])
    Ny = int(attrs["Ny"])
    dx = Lx / Nx
    dy = Ly / Ny
    Nt = data.n_snapshots

    cross_helicity = np.empty(Nt)
    for i in range(Nt):
        vx = data["vx"].frame(i)
        vy = data["vy"].frame(i)
        bx = data["bx"].frame(i)
        by = data["by"].frame(i)

        cross_helicity[i] = dx * dy * (vx * bx + vy * by).sum()

    return cross_helicity


def verify_energy(data):
    """Check energy conservation for an opened run (see `utils.lazy_fields.open_fields`).

    Frames are read lazily and processed one snapshot at a time, so peak memory
    stays at a few (Ny, Nx) arrays regardless of the number of snapshots.
    """
    attrs = data.attrs
    nu = attrs["nu"]
    eta = attrs["eta"]
    Nx = int(attrs["Nx"])
    Ny = int(attrs["Ny"])
    Lx = attrs["Lx"]
    Ly = attrs["Ly"]
    dx = Lx / Nx
    dy = Ly / Ny

    # rfft axis (x) is last, full fft axis (y) is first
    kx = data.reader.f["grid/kx"][:][None, :]
    ky = data.reader.f["grid/ky"][:][:, None]
    k2 = kx**2 + ky**2

    # Snapshot times. Spacing is nt_save * dt, not the solver dt.
    times = np.asarray(data.times, dtype=np.float64)
    Nt = data.n_snapshots

    kinetic_energy = np.empty(Nt)
    magnetic_energy = np.empty(Nt)
    intdxdy_dkedt = np.empty(Nt)
    intdxdy_dbedt = np.empty(Nt)

    for i in range(Nt):
        vx = data["vx"].frame(i)
        vy = data["vy"].frame(i)
        bx = data["bx"].frame(i)
        by = data["by"].frame(i)

        (
            kinetic_energy[i],
            magnetic_energy[i],
            intdxdy_dkedt[i],
            intdxdy_dbedt[i],
        ) = _frame_energy_terms(vx, vy, bx, by, kx, ky, k2, nu, eta, Nx, Ny, dx, dy)

    total_energy = kinetic_energy + magnetic_energy
    intdxdy_dEdt = intdxdy_dkedt + intdxdy_dbedt

    # Integrate each rate up to every snapshot time (length Nt - 1).
    trap_dkedt = cumulative_trapezoid(intdxdy_dkedt, x=times)
    intt_dkedt = kinetic_energy[0] + trap_dkedt

    trap_dbedt = cumulative_trapezoid(intdxdy_dbedt, x=times)
    intt_dbedt = magnetic_energy[0] + trap_dbedt

    trap_dEdt = cumulative_trapezoid(intdxdy_dEdt, x=times)
    intt_dEdt = total_energy[0] + trap_dEdt

    num_dkedt = np.gradient(kinetic_energy, times, edge_order=2)
    num_dbedt = np.gradient(magnetic_energy, times, edge_order=2)
    num_dEdt = np.gradient(total_energy, times, edge_order=2)

    # cumulative_trapezoid (no `initial`) returns length N-1, where element i is the
    # integral from t0 to t_{i+1}. Adding E[0] yields the predicted energy at
    # t_{i+1}, so compare against E[1:] (not E[:-1]) to stay time-aligned.
    ke_err = np.abs(kinetic_energy[1:] - intt_dkedt)
    me_err = np.abs(magnetic_energy[1:] - intt_dbedt)
    ed_err = np.abs(total_energy[1:] - intt_dEdt)

    cross_hel = cross_helicity(data)
    mag_hel = mag_helicity(data)

    cross_hel_err = (cross_hel - cross_hel[0]) / cross_hel[0]
    tot_energy_err_rel = (total_energy - total_energy[0]) / total_energy[0]

    return {
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
        "num_dEdt": num_dEdt,
        "dEdt": intdxdy_dEdt,
        "trap_dkedt": trap_dkedt,
        "int_dEdt": intt_dEdt,
        "cross_helicity_err": cross_hel_err,
        "mag_helicity": mag_hel,
        "tot_energy_err_rel": tot_energy_err_rel,
        "t": times,
        "dt": float(np.median(np.diff(times))),
    }


def alfven_velocity(data, n_lines=5):
    """Velocity amplitude diagnostics for an Alfven wave travelling in x.

    Returns max |vx|, max |vy| and the rms speed at every snapshot, plus vy
    line-outs along x (at the middle row in y) for `n_lines` evenly spaced
    snapshots. Also returns the linear-theory decay rate of the dominant initial
    mode so the measured amplitude can be compared against an envelope.
    """
    Nt = data.n_snapshots
    attrs = data.attrs
    Nx = int(attrs["Nx"])
    Ny = int(attrs["Ny"])
    row = Ny // 2

    # Dominant (non-mean) Fourier mode of the initial velocity gives the wave's k.
    kx_1d = data.reader.f["grid/kx"][:]
    ky_1d = data.reader.f["grid/ky"][:]
    power = np.abs(fft2d(data["vx"].frame(0), Nx, Ny)) ** 2
    power += np.abs(fft2d(data["vy"].frame(0), Nx, Ny)) ** 2
    power[0, 0] = 0.0
    j, i = np.unravel_index(np.argmax(power), power.shape)
    kx_dom, ky_dom = kx_1d[i], ky_1d[j]
    k2_dom = kx_dom**2 + ky_dom**2

    nu, eta = attrs["nu"], attrs["eta"]
    # k_dot_b0 = kx_dom * attrs.get("b0x", 0.0) + ky_dom * attrs.get("b0y", 0.0)
    # disc = (0.5 * (nu - eta) * k2_dom) ** 2 - k_dot_b0**2
    decay_rate = 0.5 * (nu + eta) * k2_dom  # - np.real(np.sqrt(disc + 0j))
    line_idx = np.unique(np.linspace(0, Nt - 1, min(n_lines, Nt)).astype(int))

    vx_max = np.empty(Nt)
    vy_max = np.empty(Nt)
    v_rms = np.empty(Nt)
    lineouts = {}

    for i in range(Nt):
        vx = data["vx"].frame(i)
        vy = data["vy"].frame(i)
        vx_max[i] = np.max(np.abs(vx))
        vy_max[i] = np.max(np.abs(vy))
        v_rms[i] = np.sqrt(np.mean(vx**2 + vy**2))
        if i in line_idx:
            lineouts[i] = vy[row].copy()

    return {
        "t": np.asarray(data.times, dtype=np.float64),
        "x": np.asarray(data.x),
        "y_row": float(np.asarray(data.y)[row]),
        "vx_max": vx_max,
        "vy_max": vy_max,
        "v_rms": v_rms,
        "lineouts": lineouts,
        "kx_dom": float(kx_dom),
        "ky_dom": float(ky_dom),
        "decay_rate": float(decay_rate),
    }


def show_alfven_plots(alfven_result, save_path: str | Path = "."):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    t = alfven_result["t"]
    x = alfven_result["x"]

    rate = alfven_result["decay_rate"]
    envelope = np.exp(-rate * (t - t[0]))
    env_label = (
        rf"theory: $e^{{-\gamma t}}$, $\gamma$={rate:.3g} "
        rf"($k_x$={alfven_result['kx_dom']:.3g}, $k_y$={alfven_result['ky_dom']:.3g})"
    )
    env_style = {"color": "k", "marker": "", "linestyle": "--", "linewidth": 2}

    fig, ax = plt.subplots(3, 1, figsize=(15, 20))

    # Amplitude over time, with the linear-theory decay envelope.
    ax[0].plot(t, alfven_result["vy_max"], label=r"max $|v_y|$", alpha=0.5)
    ax[0].plot(t, alfven_result["vx_max"], label=r"max $|v_x|$", alpha=0.5)
    ax[0].plot(t, alfven_result["v_rms"], label="rms speed", alpha=0.5)
    ax[0].plot(t, alfven_result["vy_max"][0] * envelope, label=env_label, **env_style)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("Velocity amplitude")
    ax[0].legend()

    # Same thing normalized by the initial value, to read off fractional change.
    for key, label in (
        ("vy_max", r"max $|v_y|$"),
        ("vx_max", r"max $|v_x|$"),
        ("v_rms", "rms speed"),
    ):
        v0 = alfven_result[key][0]
        if v0 != 0:
            ax[1].plot(t, alfven_result[key] / v0, label=label, alpha=0.5)
    ax[1].plot(t, envelope, label=env_label, **env_style)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("Amplitude / initial amplitude")
    ax[1].legend()

    # vy along x at a fixed y for a few times, to see the wave travel.
    for i, line in alfven_result["lineouts"].items():
        ax[2].plot(
            x, line, label=f"t = {t[i]:.4g}", alpha=0.7, marker="", linestyle="-"
        )
    ax[2].set_xlabel("x")
    ax[2].set_ylabel(r"$v_y$")
    ax[2].set_title(f"$v_y$ along x at y = {alfven_result['y_row']:.4g}")
    ax[2].legend()

    fig.tight_layout()
    fig.savefig(save_path / "alfven_velocity.png")
    plt.close(fig)


def show_energy_plots(energy_result, save_path: str | Path = "."):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    plt.close("all")
    fig, ax = plt.subplots(5, 2, figsize=(15, 22))
    t = energy_result["t"]
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
    ax2[0].set_title(f"Error with snapshot spacing ({energy_result['dt']})")
    ax2[1].set_title(f"Error with snapshot spacing ({energy_result['dt']})")

    plt.tight_layout()
    fig2.tight_layout()
    fig.savefig(save_path / "energy.png")
    fig2.savefig(save_path / "energy_error.png")
    plt.close(fig)
    plt.close(fig2)


def show_total_energy_plots(energy_result, save_path: str | Path = "."):
    """Total-energy counterpart of the kinetic/magnetic plots in `show_energy_plots`."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    t = energy_result["t"]
    deriv_error = np.abs(energy_result["num_dEdt"] - energy_result["dEdt"])[:-2]

    fig, ax = plt.subplots(5, 1, figsize=(10, 22))
    # Row 0: raw total energy
    ax[0].plot(t[:-1], energy_result["ed"][:-1], label="Total Energy", alpha=0.5)
    ax[0].legend()

    # Row 1: energy integrated in time
    ax[1].plot(
        t[:-1],
        energy_result["int_dEdt"],
        label="Total Energy (Integrated in time)",
        alpha=0.5,
    )
    ax[1].legend()

    # Row 2: dE/dt (numerical derivative)
    ax[2].plot(
        t[:-2],
        energy_result["num_dEdt"][:-2],
        label="Total Energy Derivative",
        alpha=0.5,
    )
    ax[2].legend()

    # Row 3: dE/dt integrated in space
    ax[3].plot(
        t[:-1],
        energy_result["dEdt"][:-1],
        label="Total Energy (dE/dt integrated in space)",
        alpha=0.5,
    )
    ax[3].legend()

    # Row 4: error (log scale)
    ax[4].semilogy(
        t[:-2], energy_result["ed_err"][:-1], label="Total Energy Error", alpha=0.5
    )
    ax[4].semilogy(
        t[:-2], deriv_error, label="Total Energy Derivative Error", alpha=0.5
    )
    ax[4].legend()

    fig2, ax2 = plt.subplots(1, 1, figsize=(15, 8))
    ax2.plot(
        t[:-2], energy_result["ed_err"][:-1], label="Total Energy Error", alpha=0.5
    )
    ax2.plot(t[:-2], deriv_error, label="Total Energy Derivative Error", alpha=0.5)
    ax2.legend()
    ax2.set_title(f"Error with snapshot spacing ({energy_result['dt']})")

    fig.tight_layout()
    fig2.tight_layout()
    fig.savefig(save_path / "energy_total.png")
    fig2.savefig(save_path / "energy_total_error.png")
    plt.close(fig)
    plt.close(fig2)


def show_conservation_error_plots(result, save_path: str | Path = "."):
    """Three stacked panels: total energy error, cross helicity error, magnetic helicity."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    t = result["t"]

    panels = [
        ("tot_energy_err_rel", r"$(E(t) - E(0)) / E(0)$"),
        ("cross_helicity_err", r"$(H_c(t) - H_c(0)) / H_c(0)$"),
        ("mag_helicity", r"$H_m(t)$"),
    ]
    fig, ax = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    for a, (key, label) in zip(ax, panels):
        a.plot(t, result[key], alpha=0.5)
        a.set_ylabel(label)
    ax[-1].set_xlabel("t")

    fig.tight_layout()
    fig.savefig(save_path / "conservation_errors.png")
    plt.close(fig)


def main():
    args = parse_args()
    run_file = resolve_run_file(args.data_path)
    save_path = Path(args.output_path) if args.output_path else run_file.parent / "viz"

    with open_fields(run_file) as data:
        attrs = data.attrs
        if not attrs.get("completed", True):
            print(f"WARNING: run did not complete: {attrs.get('error', 'unknown')}")
        if attrs.get("hyper", False):
            print(
                "WARNING: run used hyperviscosity; the nu/eta Laplacian dissipation "
                "terms here will not match the solver."
            )
        print(
            f"{run_file}: {data.n_snapshots} snapshots, "
            f"{attrs['Ny']}x{attrs['Nx']}, "
            f"t = {data.times[0]:.4g} to {data.times[-1]:.4g}"
        )
        result = verify_energy(data)
        alfven_result = alfven_velocity(data) if args.alfven else None

    print(save_path)
    show_energy_plots(result, save_path=save_path)
    show_total_energy_plots(result, save_path=save_path)
    show_conservation_error_plots(result, save_path=save_path)
    if alfven_result is not None:
        show_alfven_plots(alfven_result, save_path=save_path)


if __name__ == "__main__":
    main()
