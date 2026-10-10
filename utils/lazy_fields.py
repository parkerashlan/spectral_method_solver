"""
Lazy adapter between RunReader and AnimatePhysics so that nothing
is held in memory except a small cache of recent frames. Helps keep
the peak memory to a few 2D arrays.
"""

from functools import lru_cache

import numpy as np


class LazyField:
    """
    Gives one derived fields as if it were a (Ny, Nx, n_snap) array.
    """

    def __init__(self, run, name, transform=None):
        self._source = run
        self.name = name
        self._transform = transform

        ny, nx = run.frame_shape
        self.shape = (ny, nx, run.n_snapshots)
        self.ndim = 3
        self.dtype = np.dtype("f8")

        self._range = None

    def frame(self, t):
        """Return 2D field at index t."""
        if t < 0:
            t += self.shape[2]

        arr = self._source.get(self.name, t)
        if self._transform is not None:
            arr = self._transform(arr)
        return arr

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 3:
            raise TypeError(f"{type(self).__name__} supports [:, :, t] indexing only")

        ky, kx, kt = key
        if not (isinstance(ky, slice) and isinstance(kx, slice)):
            raise TypeError("The first two indices must be full slices")

        if isinstance(kt, slice):
            raise TypeError("Slicing the time axis would load all frames into memory.")

        return self.frame(int(kt))

    def __len__(self):
        return self.shape[0]

    def value_range(self, sample=None):
        """
        Global (min, max) over time from scanning all frames.

        sample: int or None
            Scan this many evenly spaced snapshots. None scans all frames.
        """

        if self._range is not None:
            return self._range

        n = self.shape[2]
        idx = (
            range(n)
            if sample is None
            else np.unique(
                np.linspace(0, n - 1, min(sample, n)).astype(int).astype(int)
            )
        )

        lo, hi = np.inf, -np.inf
        for t in idx:
            f = self.frame(int(t))
            lo = min(lo, np.nanmin(f))
            hi = max(hi, np.nanmax(f))
        self._range = (float(lo), float(hi))
        return self._range

    @property
    def min(self):
        return self.value_range()[0]

    @property
    def max(self):
        return self.value_range()[1]

    def __array__(self):
        raise MemoryError(
            f"Refuse to render {self.name!r}: that is "
            f"{np.prod(self.shape) * 8 / 2**30} GB. "
            f"Use .frame(t) or value_range() for limits."
        )


class LazyRun:
    """
    Dict-like view over a run file. Keys are field names, values are LazyField objects.
    """

    def __init__(self, reader, cache_size=4):
        self.reader = reader
        self.attrs = reader.attrs
        self.frame_shape = reader.A.shape[1:]
        self.n_snapshots = reader.n_snapshots
        self.times = reader.t[:]

        self.x = reader.f["grid/x"][:]
        self.y = reader.f["grid/y"][:]

        # Reader declares the what methods produce which fields
        self._spec = dict(reader.FIELDS)

        # One cache per source method since a method could have multiple
        # fields
        self._cached = {
            method: lru_cache(cache_size)(getattr(reader, method))
            for method in {m for m, _ in self._spec.values()}
        }

        # The self being passed to LazyField is the reader itself
        self._fields = {name: LazyField(self, name) for name in self._spec}

    def get(self, name, t):
        """Return value of field at time t."""
        method, return_idx = self._spec[name]
        out = self._cached[method](t)
        return out if return_idx is None else out[return_idx]

    def __getitem__(self, name):
        """Return LazyField object for given field"""
        return self._fields[name]

    def __setitem__(self, key, value):
        self._fields[key] = value

    def __contains__(self, key):
        return key in self._fields

    def keys(self):
        return self._fields.keys()

    def values(self):
        return self._fields.values()

    def items(self):
        return self._fields.items()

    def add_derived(self, name, source_key, transform):
        """
        Register a field computed per-frame from another.

        data.add_derived("j2", "jz", lambda a: a**2) would add j^2 as a derived field.
        """

        field = LazyField(self, source_key, transform=transform)
        self._fields[name] = field
        return field

    def close(self):
        self.reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


def open_fields(path, cache_size=4):
    """Open a run file and return a LazyRun"""

    from solver.io_hdf5 import RunReader

    return LazyRun(RunReader(path), cache_size=cache_size)
