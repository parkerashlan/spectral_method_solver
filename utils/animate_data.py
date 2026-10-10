from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize


def _value_range(data, sample=None):
    """(min, max) over time, without materialising a lazy field."""
    if hasattr(data, "value_range"):
        return data.value_range(sample=sample)
    return float(np.nanmin(data)), float(np.nanmax(data))


def _speed_range(vx, vy, sample=None):
    """(min, max) of sqrt(vx^2 + vy^2) over time."""
    if not hasattr(vx, "frame"):
        s = np.sqrt(vx**2 + vy**2)
        return float(np.nanmin(s)), float(np.nanmax(s))

    n = vx.shape[2]
    idx = (
        range(n)
        if sample is None
        else np.unique(np.linspace(0, n - 1, min(sample, n)).astype(int))
    )
    lo, hi = np.inf, -np.inf
    for t in idx:
        s = np.sqrt(vx.frame(int(t)) ** 2 + vy.frame(int(t)) ** 2)
        lo = min(lo, np.nanmin(s))
        hi = max(hi, np.nanmax(s))
    return float(lo), float(hi)


class AnimatePhysics:
    def __init__(
        self,
        data,
        x=None,
        y=None,
        vmin=None,
        vmax=None,
        title=None,
        figsize=None,
        panel_size=6.0,
        colorbar=False,
        cmap="jet",
        nt_save=1,
        interval=50,
        rows=1,
        cols=1,
        save_path=".",
        filename="animation.mp4",
        frame_step=1,
        dt=1.0,
        times=None,
        range_sample=None,
        colorbar_kwargs=None,
    ):
        data_shape = None
        for _, value in data.items():
            shape = getattr(value, "shape", None)
            if shape is not None and len(shape) == 3:
                data_shape = shape
                break
        if data_shape is None:
            raise ValueError("no 3D field found in data")

        self.data = data
        self.x = x if x is not None else np.arange(data_shape[0])
        self.y = y if y is not None else np.arange(data_shape[1])
        self.vmin = vmin
        self.vmax = vmax
        self.title = title

        if figsize is None:
            # Size each panel from the actual data aspect ratio (y-range /
            # x-range) so that equal-aspect axes (imshow/streamplot/contour)
            # fill their subplot box instead of leaving blank space around
            # a square that doesn't match the grid of rows/cols requested.
            x_range = float(np.max(self.x) - np.min(self.x)) or 1.0
            y_range = float(np.max(self.y) - np.min(self.y)) or 1.0
            data_aspect = abs(y_range / x_range)

            panel_width = panel_size * (1.25 if colorbar else 1.0)
            fig_width = cols * panel_width
            fig_height = rows * panel_size * data_aspect
            figsize = (fig_width, fig_height)

        self.figsize = figsize
        self.colorbar = colorbar
        # Passed straight through to fig.colorbar(...). Useful keys for a
        # wide/thin plot: "fraction" (width, as a fraction of the axes size),
        # "shrink" (length along the axes, 0-1), "aspect" (long/short side
        # ratio of the colorbar itself), "pad" (gap between axes and
        # colorbar).
        self.colorbar_kwargs = colorbar_kwargs or {}

        self.cmap = cmap
        self.nt_save = nt_save
        self.nt = data_shape[-1]
        self.frame_step = frame_step
        self.interval = interval
        self.rows = rows
        self.cols = cols
        self.filename = filename
        self.save_path = save_path
        self.dt = dt
        # Actual snapshot times from the run file. Falls back to the old
        # index * nt_save * dt estimate when not supplied.
        self.times = None if times is None else np.asarray(times)
        self.range_sample = range_sample

        self.fig, self.ax = plt.subplots(
            figsize=self.figsize, nrows=rows, ncols=cols, constrained_layout=True
        )
        if rows == 1 and cols == 1:
            self.ax = np.array([[self.ax]])
        elif rows == 1:
            self.ax = self.ax[np.newaxis, :]
        elif cols == 1:
            self.ax = self.ax[:, np.newaxis]

    def save_timesteps(
        self,
        timesteps,
        data_key,
        plot_type,
        title=None,
        save_path=None,
        filename_prefix=None,
        bounds=None,
        **plot_kwargs,
    ):
        """Save individual plot images for one or more timesteps.

        Parameters
        ----------
        timesteps : float or list of float
            Timestep(s) in seconds to save. Each is converted to the
            nearest saved frame index using ``dt`` and ``nt_save``.
        data_key : str or tuple
            Key into ``self.data``. Must be a ``(vx_key, vy_key)`` tuple
            when ``plot_type`` is ``"streamplot"``.
        plot_type : str
            One of "field", "streamplot", "contour".
        title : str, optional
            Title for the plot. Ignored if ``self.title`` is set.
        save_path : str
            Directory to save the images into.
        filename_prefix : str, optional
            Prefix used for saved filenames. Defaults to ``data_key``
            (joined with "_" for streamplot tuples).
        bounds : list or tuple, optional
            ``[y0, y1, x0, x1]`` bounds (in data coordinates, i.e. the
            units of ``self.x``/``self.y``) to crop the saved plot to. If
            not given, the full extent of the data is shown.

            For "field" plots, ``bounds`` also determines the colorbar
            range: it's set from the min/max of the data *within the
            cropped region* at each saved timestep. If ``bounds`` is not
            given, the colorbar instead uses the min/max of the full
            frame at that timestep. Either way, the color range is
            computed per-timestep rather than over the entire time
            series.
        **plot_kwargs
            Extra kwargs forwarded to the plot-type-specific initializer,
            e.g. ``norm=`` for "streamplot" or ``levels=`` for "contour".
        """
        if isinstance(timesteps, (int, float)):
            timesteps = [timesteps]

        if save_path is None:
            save_path = self.save_path

        ax_coords = (0, 0)

        if filename_prefix is None:
            filename_prefix = (
                "_".join(data_key) if isinstance(data_key, tuple) else data_key
            )

        y_idx = x_idx = None
        if bounds is not None:
            y0, y1, x0, x1 = bounds
            y_idx = np.where((self.y >= y0) & (self.y <= y1))[0]
            x_idx = np.where((self.x >= x0) & (self.x <= x1))[0]
            if y_idx.size == 0 or x_idx.size == 0:
                raise ValueError(
                    f"bounds {bounds!r} (y0, y1, x0, x1) select no grid points "
                    f"(y ranges over [{self.y.min():.4g}, {self.y.max():.4g}], "
                    f"x ranges over [{self.x.min():.4g}, {self.x.max():.4g}]). "
                    "Check that y1 > y0 and x1 > x0."
                )

        if plot_type == "streamplot":
            params = self._initialize_streamplot(
                data_key, ax_coords, plot_kwargs.get("norm"), title
            )
        elif plot_type == "contour":
            params = self._initialize_contour(
                data_key, ax_coords, title, plot_kwargs.get("levels", 20)
            )
        elif plot_type != "field":
            raise ValueError(f"Unknown plot type: {plot_type}")

        colorbar_obj = None
        if plot_type != "field":
            params["ax_coords"] = ax_coords
            params["type"] = plot_type

            if self.colorbar:
                sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=params["norm"])
                sm.set_array([])
                colorbar_obj = self.fig.colorbar(
                    sm, ax=self.ax[ax_coords], **self.colorbar_kwargs
                )

        original_vmin, original_vmax = self.vmin, self.vmax

        try:
            for t in timesteps:
                if self.times is not None:
                    idx = int(np.argmin(np.abs(self.times - t)))
                else:
                    idx = round(t / (self.nt_save * self.dt))
                idx = max(0, min(idx, self.nt - 1))

                if plot_type == "field":
                    frame = self.data[data_key][:, :, idx]
                    crop = (
                        frame[np.ix_(y_idx, x_idx)]
                        if y_idx is not None and x_idx is not None
                        else frame
                    )
                    self.vmin = np.nanmin(crop)
                    self.vmax = np.nanmax(crop)

                    params = self._initialize_field(data_key, ax_coords, title)
                    params["ax_coords"] = ax_coords
                    params["type"] = plot_type

                    if self.colorbar:
                        if colorbar_obj is not None:
                            colorbar_obj.remove()
                        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=params["norm"])
                        sm.set_array([])
                        colorbar_obj = self.fig.colorbar(
                            sm, ax=self.ax[ax_coords], **self.colorbar_kwargs
                        )

                self._draw_frame(idx, [params])

                if bounds is not None:
                    self.ax[ax_coords].set_xlim(x0, x1)
                    self.ax[ax_coords].set_ylim(y0, y1)

                out_path = f"{save_path}/{filename_prefix}_t={t:.6g}.png"
                self.fig.savefig(out_path)
                print(f"Saved timestep {t}s (frame {idx}) to {out_path}")
        finally:
            self.vmin, self.vmax = original_vmin, original_vmax

    def _time_label(self, t):
        if self.times is not None and 0 <= t < len(self.times):
            return self.times[t]
        return t * self.nt_save * self.dt

    def _draw_frame(self, t, type_params):
        artists = []

        for params in type_params:
            plot_type = params["type"]
            ax_coords = params["ax_coords"]
            self.ax[ax_coords].clear()
            if plot_type == "field":
                data = params["data"]
                im = self.ax[ax_coords].imshow(
                    data[:, :, t],
                    cmap=self.cmap,
                    vmin=params["vmin"],
                    vmax=params["vmax"],
                    origin="lower",
                    # aspect="auto",
                    extent=params["extent"],
                )
                self.ax[ax_coords].set_title(
                    f"{params['title']} (t = {self._time_label(t):.4g})"
                )
            elif plot_type == "streamplot":
                vx = params["data"][0][:, :, t]
                vy = params["data"][1][:, :, t]
                color = np.sqrt(vx**2 + vy**2)
                im = self.ax[ax_coords].streamplot(
                    self.x,
                    self.y,
                    vx,
                    vy,
                    norm=params["norm"],
                    color=color,
                    cmap=self.cmap,
                    linewidth=1,
                    density=1.3,
                    broken_streamlines=False,
                )
                self.ax[ax_coords].set_title(
                    f"{params['title']} (t = {self._time_label(t):.4g})"
                )
                self.ax[ax_coords].set_aspect("equal")
            elif plot_type == "contour":
                im = self.ax[ax_coords].contourf(
                    self.x,
                    self.y,
                    params["data"][:, :, t],
                    levels=params["levels"],
                    cmap=self.cmap,
                    norm=params["norm"],
                )
                self.ax[ax_coords].contour(
                    self.x,
                    self.y,
                    params["data"][:, :, t],
                    levels=params["levels"],
                    cmap=self.cmap,
                    linewidths=0.5,
                )
                self.ax[ax_coords].set_title(
                    f"{params['title']} (t = {self._time_label(t):.4g})"
                )
            else:
                raise ValueError(f"Unknown plot type: {type}")

            artists.append((im, ax_coords, params["norm"]))

        # self.fig.tight_layout()
        return []

    def _initialize_field(self, data_key, ax_coords, title=None):
        data = self.data[data_key]

        if data.ndim != 3:
            raise ValueError('The "data" key must contain a 3D array.')

        if self.x[1] - self.x[0] < 1:
            # Add half-pixel correction since extent starts from edges of pixels but linspace gives cell centers
            dx = self.x[1] - self.x[0]
            dy = self.y[1] - self.y[0]
            extent = [
                self.x[0] - dx / 2,
                self.x[-1] + dx / 2,
                self.y[0] - dy / 2,
                self.y[-1] + dy / 2,
            ]
        else:
            extent = None

        lo, hi = _value_range(data, self.range_sample)
        vmin = self.vmin if self.vmin is not None else lo
        vmax = self.vmax if self.vmax is not None else hi
        norm = Normalize(vmin=vmin, vmax=vmax)

        im = self.ax[ax_coords].imshow(
            data[:, :, 0],
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            # aspect="auto",
            extent=extent,
        )

        field_params = {
            "first_frame": im,
            "data": data,
            "vmin": vmin,
            "vmax": vmax,
            "extent": extent,
            "norm": norm,
        }

        if self.title is not None:
            title = self.title

        field_params["title"] = title

        return field_params

    def _initialize_streamplot(self, data_key, ax_coords, norm, title=None):
        if not isinstance(data_key, tuple):
            raise TypeError(
                "data_key must be a tuple of (vecx, vecy) for a streamplot."
            )

        vx = self.data[data_key[0]]
        vy = self.data[data_key[1]]
        Ny, Nx = vx.shape[:2]

        vmin, vmax = _speed_range(vx, vy, self.range_sample)
        if norm is None:
            norm = Normalize(vmin=vmin, vmax=vmax)

        im = self.ax[ax_coords].streamplot(
            self.x,
            self.y,
            vx[:, :, 0],
            vy[:, :, 0],
            color=np.sqrt(vx[:, :, 0] ** 2 + vy[:, :, 0] ** 2),
            norm=norm,
            broken_streamlines=False,
        )

        self.ax[ax_coords].set_aspect("equal")

        stream_params = {
            "first_frame": im,
            "data": (vx, vy),
            "vmin": vmin,
            "vmax": vmax,
            "norm": norm,
        }

        if self.title is not None:
            title = self.title

        stream_params["title"] = title

        return stream_params

    def _initialize_contour(self, data_key, ax_coords, title=None, levels=20):
        data = self.data[data_key]
        vmin, vmax = _value_range(data, self.range_sample)

        levels = np.linspace(vmin, vmax, levels)
        norm = Normalize(vmin=vmin, vmax=vmax)

        im = self.ax[ax_coords].contourf(
            self.x, self.y, data[:, :, 0], levels=levels, cmap=self.cmap, norm=norm
        )
        self.ax[ax_coords].contour(
            self.x, self.y, data[:, :, 0], levels=levels, cmap=self.cmap, linewidths=0.5
        )
        self.ax[ax_coords].set_aspect("equal")

        contour_params = {
            "im": im,
            "data": data,
            "levels": levels,
            "cmap": self.cmap,
            "norm": norm,
        }

        if self.title is not None:
            title = self.title

        contour_params["title"] = title

        return contour_params

    def plot_anim(self, plots: list, save_path: str = "."):

        if len(plots) > (self.rows * self.cols):
            raise ValueError(
                "Number of plots exceeds the number of subplots. Increase rows or cols."
            )

        draw_frame_params = []
        for i, plot in enumerate(plots):
            missing_keys = {"data_key", "type"} - plot.keys()
            if missing_keys:
                raise ValueError(
                    f"These keys are required for plotting: {missing_keys}"
                )

            ax_coords = self._get_ax_coords(i)
            if plot["type"] == "field":
                field_params = self._initialize_field(
                    plot.get("data_key"), ax_coords, plot.get("title")
                )

                field_params["ax_coords"] = ax_coords
                field_params["type"] = "field"
                draw_frame_params.append(field_params)
            elif plot["type"] == "streamplot":
                stream_params = self._initialize_streamplot(
                    plot.get("data_key"), ax_coords, plot.get("norm"), plot.get("title")
                )

                stream_params["ax_coords"] = ax_coords
                stream_params["type"] = "streamplot"
                draw_frame_params.append(stream_params)
            elif plot["type"] == "contour":
                contour_params = self._initialize_contour(
                    plot.get("data_key"), ax_coords, plot.get("title")
                )

                contour_params["ax_coords"] = ax_coords
                contour_params["type"] = "contour"
                draw_frame_params.append(contour_params)
            else:
                raise ValueError(f"Unknown plot type: {type}")

        if self.colorbar:
            for draw_params in draw_frame_params:
                norm = draw_params["norm"]
                ax_coords = draw_params["ax_coords"]
                sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=norm)
                sm.set_array([])
                self.fig.colorbar(sm, ax=self.ax[ax_coords], **self.colorbar_kwargs)

        # Subsample frames by frame_step (e.g. frame_step=5 uses every 5th frame).
        # The selected indices are passed straight through to _draw_frame, so data
        # indexing and titles remain correct.
        frames = range(0, self.nt, self.frame_step)

        anim = FuncAnimation(
            self.fig,
            partial(self._draw_frame, type_params=draw_frame_params),
            frames=frames,
            interval=self.interval,
        )

        save_path = f"{save_path}/{self.filename}"
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow")
        elif save_path.endswith(".mp4"):
            anim.save(save_path, writer="ffmpeg")
        print(f"Animation saved to {save_path}")

    def _get_ax_coords(self, index):
        if index == 0:
            return (0, 0)

        row = index // self.cols
        col = index % self.cols
        return (row, col)
