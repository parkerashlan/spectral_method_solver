from turtle import color

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize


def animate_field(
    data,
    x=None,
    y=None,
    cmap="jet",
    interval=50,
    vmin=None,
    vmax=None,
    title=None,
    colorbar=True,
    figsize=(8, 6),
    save_path=None,
    nt_save=1,
):
    """
    Create an animation from a 3D array where the last dimension is time.

    Parameters
    ----------
    data : ndarray
        3D array with shape (Ny, Nx, Nt) where Nt is the number of saved frames.
    x : ndarray, optional
        1D array of x coordinates with shape (Nx,). If None, uses pixel indices.
    y : ndarray, optional
        1D array of y coordinates with shape (Ny,). If None, uses pixel indices.
    cmap : str, optional
        Colormap to use. Default is 'jet'.
    interval : int, optional
        Delay between frames in milliseconds. Default is 50.
    vmin : float, optional
        Minimum value for color scale. If None, uses global min of data.
    vmax : float, optional
        Maximum value for color scale. If None, uses global max of data.
    title : str, optional
        Title template. Use {t} for the real simulation timestep, e.g. "Vorticity t={t}".
    colorbar : bool, optional
        Whether to show a colorbar. Default is True.
    figsize : tuple, optional
        Figure size (width, height). Default is (8, 6).
    save_path : str, optional
        If provided, saves animation to this path (e.g., 'animation.gif' or 'animation.mp4').
    nt_save : int, optional
        Number of simulation timesteps between saved frames. Used to compute the
        real timestep shown in the title as frame * nt_save. Default is 1.

    Returns
    -------
    anim : FuncAnimation
        The matplotlib animation object.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected 3D array, got {data.ndim}D array")

    nt = data.shape[2]

    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    if x is not None and y is not None:
        # Add half-pixel correction since extent starts from edges of pixels
        # but linspace gives cell centers
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        extent = [x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2]
    else:
        extent = None

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        data[:, :, 0],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        aspect="auto",
        extent=extent,
    )

    if colorbar:
        plt.colorbar(im, ax=ax)

    if title:
        title_obj = ax.set_title(title.format(t=0))

    def update(frame):
        im.set_array(data[:, :, frame])

        if title:
            print(frame * nt_save)
            title_obj.set_text(title.format(t=frame * nt_save))
        return [im]

    anim = FuncAnimation(fig, update, frames=nt, interval=interval, blit=True)

    if save_path:
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow")
        else:
            anim.save(save_path, writer="ffmpeg")
        print(f"Animation saved to {save_path}")

    return anim


def animate_streamplot(
    vx,
    vy,
    x=None,
    y=None,
    pot_field=None,
    interval=50,
    density=1.5,
    color_by="magnitude",
    cmap="viridis",
    linewidth=1.5,
    title=None,
    colorbar=True,
    figsize=(8, 6),
    save_path=None,
    nt_save=1,
    multi=False,
    t=None,
):
    """
    Create an animation of a time-varying 2D velocity field using streamplots.

    Parameters
    ----------
    vx : ndarray
        3D array of x-vector with shape (Ny, Nx, Nt).
    vy : ndarray
        3D array of y-vector with shape (Ny, Nx, Nt).
    x : ndarray, optional
        1D array of x coordinates with shape (Nx,). If None, uses pixel indices.
    y : ndarray, optional
        1D array of y coordinates with shape (Ny,). If None, uses pixel indices.
    pot_field : ndarray, optional
        3D array of potential field with shape (Ny, Nx, Nt). If None, no potential field is shown.
    interval : int, optional
        Delay between frames in milliseconds. Default is 50.
    density : float, optional
        Density of streamlines. Default is 1.5.
    color_by : str, optional
        How to color streamlines. Options:
          'magnitude'    - color by |v| (default)
          'vx'       - color by x-component
          'vy'       - color by y-component
          'uniform'  - single color
    cmap : str, optional
        Colormap to use. Default is 'viridis'.
    linewidth : float, optional
        Base linewidth for streamlines. Default is 1.5.
    title : str, optional
        Title template. Use {t} for the real simulation timestep, e.g. "Stream function t={t}".
    colorbar : bool, optional
        Whether to show a colorbar. Default is True.
    figsize : tuple, optional
        Figure size (width, height). Default is (8, 6).
    save_path : str, optional
        If provided, saves animation to this path (e.g., 'animation.gif' or 'animation.mp4').
    nt_save : int, optional
        Number of simulation timesteps between saved frames. Used to compute the
        real timestep shown in the title as frame * nt_save. Default is 1.
    multi : bool, optional
        If True, animate multiple plots in a single figure. Default is False.

    Returns
    -------
    anim : FuncAnimation
        The matplotlib animation object.
    """
    if vx.ndim != 3 or vy.ndim != 3:
        raise ValueError(f"Expected 3D arrays, got vx:{vx.ndim}D vy:{vy.ndim}D")
    if vx.shape != vy.shape:
        raise ValueError(
            f"vx and vy must have the same shape, got {vx.shape} vs {vy.shape}"
        )

    Ny, Nx, Nt = vx.shape

    if x is None:
        x = np.arange(Nx, dtype=float)
    if y is None:
        y = np.arange(Ny, dtype=float)

    # Helper function to grab array values to color the plot
    def _color_field(t):
        if color_by == "magnitude":
            return np.sqrt(vx[:, :, t] ** 2 + vy[:, :, t] ** 2)
        elif color_by == "vx":
            return vx[:, :, t]
        elif color_by == "vy":
            return vy[:, :, t]
        elif color_by == "uniform":
            return None
        else:
            raise ValueError(
                f"Unknown color_by value: '{color_by}'. "
                "Choose from 'magnitude', 'vx', 'vy', 'uniform'."
            )

    # Pre-compute global color normalization across all frames
    # Finds the min/max of the arrays across all timesteps
    if color_by != "uniform":
        if color_by == "magnitude":
            all_colors = np.sqrt(vx**2 + vy**2)
        elif color_by == "vx":
            all_colors = vx
        elif color_by == "vy":
            all_colors = vy
        norm = Normalize(vmin=np.nanmin(all_colors), vmax=np.nanmax(all_colors))

        if pot_field is not None:
            vmin = np.nanmin(pot_field)
            vmax = np.nanmax(pot_field)
            pot_norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None

    fig, ax = plt.subplots(figsize=figsize)

    def _draw_frame(t):
        ax.cla()

        # Get array for timestep to color based on those values
        c = _color_field(t)
        if c is not None:
            if pot_field is not None:
                ax.contourf(
                    x,
                    y,
                    pot_field[:, :, t],
                    levels=20,
                    cmap=cmap,
                    norm=pot_norm,
                    alpha=0.9,
                )
            ax.streamplot(
                x,
                y,
                vx[:, :, t],
                vy[:, :, t],
                color=c,
                cmap=cmap,
                norm=norm,
                density=density,
                linewidth=linewidth,
                broken_streamlines=False,
            )
        else:
            ax.streamplot(
                x,
                y,
                vx[:, :, t],
                vy[:, :, t],
                density=density,
                linewidth=linewidth,
                broken_streamlines=False,
            )

        ax.set_aspect("equal")

        if title:
            ax.set_title(title.format(t=t * nt_save))

    # Draw first frame and attach colorbar once
    _draw_frame(0)

    if colorbar and color_by != "uniform":
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=color_by)

    if multi:
        return _draw_frame(0)

    def update(frame):
        _draw_frame(frame)
        return []

    anim = FuncAnimation(fig, update, frames=Nt, interval=interval, blit=False)

    if save_path:
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow")
        else:
            anim.save(save_path, writer="ffmpeg")
        print(f"Animation saved to {save_path}")

    return anim


def animate_quiver(
    vx,
    vy,
    x=None,
    y=None,
    interval=50,
    stride=15,
    scale=None,
    color_by="magnitude",
    cmap="viridis",
    title=None,
    colorbar=True,
    figsize=(8, 6),
    save_path=None,
    nt_save=1,
):
    """
    Create an animation of a time-varying 2D velocity field using a quiver plot.

    Because quiver arrows can be updated in-place via set_UVC(), this animation
    uses blit=True and is faster to render than the streamplot equivalent.

    Parameters
    ----------
    vx : ndarray
        3D array of x- with shape (Ny, Nx, Nt).
    vy : ndarray
        3D array of y-vector with shape (Ny, Nx, Nt).
    x : ndarray, optional
        1D array of x coordinates with shape (Nx,). If None, uses pixel indices.
    y : ndarray, optional
        1D array of y coordinates with shape (Ny,). If None, uses pixel indices.
    interval : int, optional
        Delay between frames in milliseconds. Default is 50.
    stride : int, optional
        Subsample the grid every `stride` points in each direction to avoid
        arrow clutter. Default is 2. Set to 1 to plot every point.
    scale : float, optional
        Scales the arrow lengths. Larger values make arrows shorter. If None,
        matplotlib chooses automatically based on the first frame.
    color_by : str, optional
        How to color arrows. Options:
          'magnitude'- color by |v| (default)
          'vx'       - color by x-component
          'vy'       - color by y-component
          'uniform'  - single color (default matplotlib arrow color)
    cmap : str, optional
        Colormap to use when color_by != 'uniform'. Default is 'viridis'.
    title : str, optional
        Title template. Use {t} for the real simulation timestep, e.g. "Velocity t={t}".
    colorbar : bool, optional
        Whether to show a colorbar. Default is True.
    figsize : tuple, optional
        Figure size (width, height). Default is (8, 6).
    save_path : str, optional
        If provided, saves animation to this path (e.g., 'animation.gif' or 'animation.mp4').
    nt_save : int, optional
        Number of simulation timesteps between saved frames. Used to compute the
        real timestep shown in the title as frame * nt_save. Default is 1.

    Returns
    -------
    anim : FuncAnimation
        The matplotlib animation object.
    """
    if vx.ndim != 3 or vy.ndim != 3:
        raise ValueError(f"Expected 3D arrays, got vx:{vx.ndim}D vy:{vy.ndim}D")
    if vx.shape != vy.shape:
        raise ValueError(
            f"vx and vy must have the same shape, got {vx.shape} vs {vy.shape}"
        )

    Ny, Nx, Nt = vx.shape

    if x is None:
        x = np.arange(Nx, dtype=float)
    if y is None:
        y = np.arange(Ny, dtype=float)

    # Subsample grid by stride
    xs = x[::stride]
    ys = y[::stride]

    def _color_field(t):
        # Grabs array from specific timestep to use array values for coloring
        u = vx[::stride, ::stride, t]
        v = vy[::stride, ::stride, t]
        if color_by == "magnitude":
            return np.sqrt(u**2 + v**2)
        elif color_by == "vx":
            return u
        elif color_by == "vy":
            return v
        elif color_by == "uniform":
            return None
        else:
            raise ValueError(
                f"Unknown color_by value: '{color_by}'. "
                "Choose from 'magnitude', 'vx', 'vy', 'uniform'."
            )

    # Pre-compute global color normalization across all frames
    if color_by != "uniform":
        if color_by == "magnitude":
            all_colors = np.sqrt(
                vx[::stride, ::stride, :] ** 2 + vy[::stride, ::stride, :] ** 2
            )
        elif color_by == "vx":
            all_colors = vx[::stride, ::stride, :]
        elif color_by == "vy":
            all_colors = vy[::stride, ::stride, :]
        norm = Normalize(vmin=np.nanmin(all_colors), vmax=np.nanmax(all_colors))
    else:
        norm = None

    fig, ax = plt.subplots(figsize=figsize)

    u0 = vx[::stride, ::stride, 0]
    v0 = vy[::stride, ::stride, 0]
    c0 = _color_field(0)

    # Draw the initial quiver; keep scale fixed after first frame.
    # Pass scalars as the C positional argument so quiver handles cmap/norm
    # mapping internally -- this lets set_array() work cleanly for updates.
    if color_by != "uniform":
        q = ax.quiver(
            xs, ys, u0, v0, c0.ravel(), cmap=cmap, norm=norm, scale=scale, angles="xy"
        )
    else:
        q = ax.quiver(xs, ys, u0, v0, scale=scale, angles="xy")

    ax.set_aspect("equal")

    if title:
        title_obj = ax.set_title(title.format(t=0))

    if colorbar and color_by != "uniform":
        # Add a colorbar with label and use ScalarMappable for color mapping the norm
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=color_by)

    def update(frame):
        u = vx[::stride, ::stride, frame]
        v = vy[::stride, ::stride, frame]
        c = _color_field(frame)

        if color_by != "uniform":
            # Updates quiver plot in place with new data instead of re-creating whole plot
            q.set_UVC(u, v)
            q.set_array(c.ravel())
        else:
            q.set_UVC(u, v)

        artists = [q]
        if title:
            title_obj.set_text(title.format(t=frame * nt_save))
            artists.append(title_obj)

        return artists

    anim = FuncAnimation(fig, update, frames=Nt, interval=interval, blit=True)

    if save_path:
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow")
        else:
            anim.save(save_path, writer="ffmpeg")
        print(f"Animation saved to {save_path}")

    return anim


def animate_contour(
    data,
    x=None,
    y=None,
    levels=20,
    filled=True,
    cmap="jet",
    interval=50,
    vmin=None,
    vmax=None,
    title=None,
    colorbar=True,
    figsize=(8, 6),
    save_path=None,
    nt_save=1,
    multi=False,
    t=None,
):
    """

    Create an animation from a 3D array using contour or filled-contour plots.

    Parameters
    ----------
    data : ndarray
        3D array with shape (Ny, Nx, Nt) where Nt is the number of saved frames.
    x : ndarray, optional
        1D array of x coordinates with shape (Nx,). If None, uses pixel indices.
    y : ndarray, optional
        1D array of y coordinates with shape (Ny,). If None, uses pixel indices.
    levels : int or array-like, optional
        Number of contour levels or explicit level values. Default is 20.
    filled : bool, optional
        If True (default), use contourf (filled contours). If False, use contour (lines).
    cmap : str, optional
        Colormap to use. Default is 'jet'.
    interval : int, optional
        Delay between frames in milliseconds. Default is 50.
    vmin : float, optional
        Minimum value for contour levels. If None, uses global min of data.
    vmax : float, optional
        Maximum value for contour levels. If None, uses global max of data.
    title : str, optional
        Title template. Use {t} for the real simulation timestep, e.g. "Pressure t={t}".
    colorbar : bool, optional
        Whether to show a colorbar. Default is True.
    figsize : tuple, optional
        Figure size (width, height). Default is (8, 6).
    save_path : str, optional
        If provided, saves animation to this path (e.g., 'animation.gif' or 'animation.mp4').
    nt_save : int, optional
        Number of simulation timesteps between saved frames. Default is 1.
    multi : bool, optional
        If True, animate multiple plots in a single figure. Default is False.

    Returns
    -------
    anim : FuncAnimation
        The matplotlib animation object.

    """

    if data.ndim != 3:
        raise ValueError(f"Expected 3D array, got {data.ndim}D array instead")

    Ny, Nx, Nt = data.shape

    if x is None:
        x = np.arange(Nx, dtype=float)
    if y is None:
        y = np.arange(Ny, dtype=float)

    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    # Build fixed level values so the color mapping is consistent across frames
    if np.isscalar(levels):
        levels = np.linspace(vmin, vmax, int(levels))

    norm = Normalize(vmin=vmin, vmax=vmax)
    contour_fn = "contourf" if filled else "contour"

    fig, ax = plt.subplots(figsize=figsize)

    def _draw_frame(t):
        ax.cla()
        cf = getattr(ax, contour_fn)(
            x, y, data[:, :, t], levels=levels, cmap=cmap, norm=norm
        )
        ax.set_aspect("equal")
        if title:
            ax.set_title(f"{title} (t={t * nt_save})")
        return cf

    if multi:
        return _draw_frame(t)

    _draw_frame(0)

    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax)

    def update(frame):
        _draw_frame(frame)
        return []

    anim = FuncAnimation(fig, update, frames=Nt, interval=interval, blit=False)

    if save_path:
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow")
        else:
            anim.save(save_path, writer="ffmpeg")
        print(f"Animation saved to {save_path}")

    return anim


def animate_multiplot(
    functions, interval=50, figsize=(8, 6), save_path=None, suptitle=None
):
    types = [f.__name__.split("_")[0] for f in functions]


def show_animation(data, x=None, y=None, **kwargs):
    """
    Convenience function to create and display an animation.

    Parameters
    ----------
    data : ndarray
        3D array with shape (Ny, Nx, Nt).
    x : ndarray, optional
        1D array of x coordinates with shape (Nx,). If None, uses pixel indices.
    y : ndarray, optional
        1D array of y coordinates with shape (Ny,). If None, uses pixel indices.
    **kwargs
        Additional arguments passed to animate_field.
    """
    anim = animate_field(data, x=x, y=y, **kwargs)
    plt.show()
    return anim


def show_streamplot(vx, vy, **kwargs):
    """
    Convenience function to create and display an animated streamplot.

    Parameters
    ----------
    vx : ndarray
        3D array of x-velocity with shape (Ny, Nx, Nt).
    vy : ndarray
        3D array of y-velocity with shape (Ny, Nx, Nt).
    **kwargs
        Additional arguments passed to animate_streamplot.
    """
    anim = animate_streamplot(vx, vy, **kwargs)
    plt.show()
    return anim


def show_quiver(vx, vy, **kwargs):
    """
    Convenience function to create and display an animated quiver plot.

    Parameters
    ----------
    vx : ndarray
        3D array of x-velocity with shape (Ny, Nx, Nt).
    vy : ndarray
        3D array of y-velocity with shape (Ny, Nx, Nt).
    **kwargs
        Additional arguments passed to animate_quiver.
    """
    anim = animate_quiver(vx, vy, **kwargs)
    plt.show()
    return anim
