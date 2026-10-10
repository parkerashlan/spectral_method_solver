import numbers
from functools import cached_property

import h5py
import hdf5plugin
import numpy as np

SCALAR_KEYS = ("t", "cfl", "e_kin", "e_mag", "jmax")
_DERIVED = (
    "k2",
    "k2_div",
    "kx_grid",
    "ky_grid",
    "x_grid",
    "y_grid",
    "res1",
    "res2",
    "visc1",
    "visc2",
    "w",
    "j",
    "dealias_mask",
)


def field(name, index=None):
    """
    Decorator for methods of RunReader to return a named plottable field.

    @field("vx", 0)
    @field("vy", 1)
    def velocity(self, i):
    Index selects a component when the method returns a tuple
    """

    def deco(fn):
        fn._fields = getattr(fn, "_fields", []) + [(name, index)]
        return fn

    return deco


def collect_fields(cls):
    """Collect all field decorators from a class and attach them as cls.FIELDS."""
    fields = {}
    for attr, obj in vars(cls).items():
        for name, index in getattr(obj, "_fields", []):
            if name in fields:
                raise ValueError(f"field {name!r} is declared twice on {cls.__name__}")
            fields[name] = (attr, index)
    cls.FIELDS = fields
    return cls


def _filter_opts(compression):
    """Return the h5py keyword arguments for the requested filter."""

    if compression is None:
        return {}
    elif compression == "gzip":
        return {"shuffle": True, "compression": "gzip", "compression_opts": 4}
    elif compression == "blosc":
        return dict(
            **hdf5plugin.Blosc2(
                cname="zstd", clevel=5, filters=hdf5plugin.Blosc2.BITSHUFFLE
            )
        )
    else:
        raise ValueError(f"Unknown compression: {compression}")


class RunWriter:
    """
    Writer for simulation runs. Will be a context manager so that the file
    closes even if the run fails.

    Example usage:
        with RunWriter("run.h5", params) as out:
            for t in range(nt):
                ...
                out.write_scalars(t, cfl=..., jmax=...)
                if t % nt_save == 0:
                    out.append_fields(t * dt, A_phys, psi_phys)
    """

    def __init__(self, path, params, compression="gzip", flush_every=10):
        """
        Initialize the RunWriter.

        Parameters
        ----------
        path : str
            Path to the output HDF5 file.
        params : dict
            Solver parameter object.
        compression : str, optional
            Compression algorithm to use. Default is "gzip".
        flush_every : int, optional
            Number of iterations between flushing to disk. Default is 10.
        """

        self.path = str(path)
        self._flush_every = int(flush_every)
        self._n_written = 0

        ny, nx = int(params.Ny), int(params.Nx)
        nt = int(params.nt)

        self.f = h5py.File(self.path, "w")

        self._record_params(params)

        fg = self.f.create_group("fields")
        opts = {
            "shape": (0, ny, nx),
            "maxshape": (None, ny, nx),
            "dtype": "f8",
            "chunks": (1, ny, nx),
            **_filter_opts(compression),
        }
        self._A_ds = fg.create_dataset("A", **opts)
        self._psi_ds = fg.create_dataset("psi", **opts)
        self._dt_snap = fg.create_dataset("t", shape=(0,), maxshape=(None,), dtype="f8")
        self._A_ds.attrs["long_name"] = "Vector potential"
        self._psi_ds.attrs["long_name"] = "Stream function"

        sg = self.f.create_group("scalars")
        self._scal = {
            k: sg.create_dataset(k, shape=(nt,), dtype="f8", fillvalue=np.nan)
            for k in SCALAR_KEYS
        }

    def _record_params(self, params, max_elements=1 << 18):
        g = self.f.create_group("grid")
        skipped = []

        for varname, value in params:
            if varname.startswith("_") or callable(value) or varname in _DERIVED:
                skipped.append(varname)
                continue

            if isinstance(value, (str, numbers.Number)):
                self.f.attrs[varname] = value
                continue

            if isinstance(value, np.ndarray):
                if value.dtype == object or value.size > max_elements:
                    skipped.append(varname)
                else:
                    g.create_dataset(varname, data=value)
                continue

            skipped.append(varname)

        self.f.attrs["skipped_params"] = ",".join(skipped) or "none"

    def append_fields(self, time, A, psi):
        """Append one snapshot of the fields to the file."""

        for ds, arr in ((self._A_ds, A), (self._psi_ds, psi)):
            # Resize the dataset to accommodate the new data
            # Do this every snapshot in case run fails
            ds.resize(ds.shape[0] + 1, axis=0)
            ds[-1] = arr

        self._dt_snap.resize(self._dt_snap.shape[0] + 1, axis=0)
        self._dt_snap[-1] = float(time)

        self._n_written += 1
        if self._n_written % self._flush_every == 0:
            self.f.flush()

    def write_scalars(self, step, **values):
        """
        Write per-step diagnostics every step.

        Unknown keys will raise a `KeyError`.
        """

        for key, val in values.items():
            if key not in self._scal:
                raise KeyError(f"{key!r} is not in SCALAR_KEYS: {SCALAR_KEYS}")
            self._scal[key][step] = float(val)

    def close(self):
        if self.f:
            self.f.flush()
            self.f.close()
            self.f = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # Record how the run ended then close
        if self.f:
            self.f.attrs["completed"] = exc_type is None
            if exc_type is not None:
                self.f.attrs["error"] = f"{exc_type.__name__}: {exc}"
        self.close()
        return False


@collect_fields
class RunReader:
    """
    Read a run file and rebuild the derived fields.

    Example Usage:
        with RunReader("run.h5") as r:
            print(r.n_snapshots, r.attrs["dt"])
            jz = r.current_density(-1)      # last snapshot
            e  = r.scalar("e_mag")
    """

    def __init__(self, path):
        self.f = h5py.File(str(path), "r")
        self.attrs = dict(self.f.attrs)
        self.A = self.f["fields/A"]
        self.psi = self.f["fields/psi"]
        self.t = self.f["fields/t"]

        self._kx = self.f["grid/kx"][:][None, :]  # rfft axis, last
        self._ky = self.f["grid/ky"][:][:, None]  # full fft axis, first
        self._k2 = (
            self._kx**2 + self._ky**2
        )  # reshaping the arrays allow for broadcasting

    @property
    def n_snapshots(self):
        return self.A.shape[0]

    def scalar(self, name):
        """Return one per-step diagnostic, with unreached steps trimmed."""
        a = self.f["scalars"][name][:]
        good = ~np.isnan(a)  # may mask errors
        return a[good]

    def _lap(self, field):
        """Take the Laplacian of a field using the spectral method."""
        fh = np.fft.rfft2(field)
        return np.fft.irfft2(-self._k2 * fh, s=field.shape)

    def _grad(self, field):
        """Take the gradient of a field using the spectral method."""
        fh = np.fft.rfft2(field)
        dx = np.fft.irfft2(1j * self._kx * fh, s=field.shape)
        dy = np.fft.irfft2(1j * self._ky * fh, s=field.shape)
        return dx, dy

    @field("A")
    def vector_potential(self, i):
        """Return the vector potential at snapshot i."""
        return self.A[i]

    @field("psi")
    def stream_function(self, i):
        """Return the stream function at snapshot i."""
        return self.psi[i]

    @field("jz")
    def current_density(self, i):
        """Return the current density at snapshot i."""
        return -self._lap(self.A[i])

    @field("vorticity")
    def vorticity(self, i):
        """Return the vorticity at snapshot i."""
        return -self._lap(self.psi[i])

    @field("bx", 0)
    @field("by", 1)
    def magnetic_field(self, i):
        """Return the magnetic field at snapshot i."""
        dAdx, dAdy = self._grad(self.A[i])
        return dAdy + self.attrs["b0x"], -dAdx + self.attrs["b0y"]

    @field("vx", 0)
    @field("vy", 1)
    def velocity(self, i):
        """Return the velocity at snapshot i."""
        dpdx, dpdy = self._grad(self.psi[i])
        return dpdy, -dpdx

    @cached_property
    def dealias_mask(self):
        """Return the dealias mask at snapshot i."""
        return (np.abs(self._kx) <= self.attrs["kx_cut"]) & (
            np.abs(self._ky) <= self.attrs["ky_cut"]
        )

    def close(self):
        if self.f:
            self.f.close()
            self.f = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False
