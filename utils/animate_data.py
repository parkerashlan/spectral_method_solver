from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize


class AnimatePhysics:
    def __init__(
        self,
        data,
        x=None,
        y=None,
        vmin=None,
        vmax=None,
        title=None,
        figsize=(15, 15),
        colorbar=False,
        cmap="jet",
        nt_save=1,
        interval=50,
        rows=1,
        cols=1,
        filename="animation.mp4",
        frame_step=1,
    ):
        for _, value in data.items():
            if isinstance(value, np.ndarray):
                data_shape = value.shape
                break

        self.data = data
        self.x = x if x is not None else np.arange(data_shape[0])
        self.y = y if y is not None else np.arange(data_shape[1])
        self.vmin = vmin
        self.vmax = vmax
        self.title = title
        self.figsize = figsize
        self.colorbar = colorbar
        self.cmap = cmap
        self.nt_save = nt_save
        self.nt = data_shape[-1]
        self.frame_step = frame_step
        self.interval = interval
        self.rows = rows
        self.cols = cols
        self.filename = filename

        self.fig, self.ax = plt.subplots(
            figsize=self.figsize, nrows=rows, ncols=cols, constrained_layout=True
        )
        if rows == 1 and cols == 1:
            self.ax = np.array([[self.ax]])
        elif rows == 1:
            self.ax = self.ax[np.newaxis, :]
        elif cols == 1:
            self.ax = self.ax[:, np.newaxis]

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
                    aspect="auto",
                    extent=params["extent"],
                )
                self.ax[ax_coords].set_title(
                    f"{params['title']} (t = {t * self.nt_save})"
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
                    density=1.5,
                    broken_streamlines=False,
                )
                self.ax[ax_coords].set_title(
                    f"{params['title']} (t = {t * self.nt_save})"
                )
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
                    f"{params['title']} (t = {t * self.nt_save})"
                )
                artists.append((im, ax_coords, params["norm"]))
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

        vmin = self.vmin if self.vmin is not None else np.nanmin(data)
        vmax = self.vmax if self.vmax is not None else np.nanmax(data)
        norm = Normalize(vmin=vmin, vmax=vmax)

        im = self.ax[ax_coords].imshow(
            data[:, :, 0],
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            aspect="auto",
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
            raise ValueError(
                "data_key must be a tuple of (vecx, vecy) for a streamplot."
            )

        vx = self.data[data_key[0]]
        vy = self.data[data_key[1]]
        Ny, Nx = vx.shape[:2]

        vmin = np.nanmin(np.sqrt(vx**2 + vy**2))
        vmax = np.nanmax(np.sqrt(vx**2 + vy**2))
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
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)

        levels = np.linspace(vmin, vmax, levels)
        norm = Normalize(vmin=vmin, vmax=vmax)

        im = self.ax[ax_coords].contourf(
            self.x, self.y, data[:, :0], levels=levels, cmap=self.cmap, norm=norm
        )
        self.ax[ax_coords].contour(
            self.x, self.y, data[:, :0], levels=levels, cmap=self.cmap, linewidths=0.5
        )
        self.ax[ax_coords].set_aspect("equal")

        contour_params = {
            "im": im,
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
                self.fig.colorbar(sm, ax=self.ax[ax_coords])

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
