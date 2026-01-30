import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_field(data, cmap='jet', interval=50, vmin=None, vmax=None, 
                  title=None, colorbar=True, figsize=(8, 6), save_path=None):
    """
    Create an animation from a 3D array where the last dimension is time.
    
    Parameters
    ----------
    data : ndarray
        3D array with shape (Nx, Ny, Nt) where Nt is the number of time steps.
    cmap : str, optional
        Colormap to use. Default is 'jet'.
    interval : int, optional
        Delay between frames in milliseconds. Default is 50.
    vmin : float, optional
        Minimum value for color scale. If None, uses global min of data.
    vmax : float, optional
        Maximum value for color scale. If None, uses global max of data.
    title : str, optional
        Title template. Use {t} for frame index, e.g. "Time step: {t}".
    colorbar : bool, optional
        Whether to show a colorbar. Default is True.
    figsize : tuple, optional
        Figure size (width, height). Default is (8, 6).
    save_path : str, optional
        If provided, saves animation to this path (e.g., 'animation.gif' or 'animation.mp4').
        
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
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data[:, :, 0], cmap=cmap, vmin=vmin, vmax=vmax, 
                   origin='lower', aspect='auto')
    
    if colorbar:
        plt.colorbar(im, ax=ax)
    
    if title:
        ax.set_title(title.format(t=0))
    
    def update(frame):
        im.set_array(data[:, :, frame])
        if title:
            ax.set_title(title.format(t=frame))
        return [im]
    
    anim = FuncAnimation(fig, update, frames=nt, interval=interval, blit=True)
    
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow')
        else:
            anim.save(save_path, writer='ffmpeg')
        print(f"Animation saved to {save_path}")
    
    return anim


def show_animation(data, **kwargs):
    """
    Convenience function to create and display an animation.
    
    Parameters
    ----------
    data : ndarray
        3D array with shape (Nx, Ny, Nt).
    **kwargs
        Additional arguments passed to animate_field.
    """
    anim = animate_field(data, **kwargs)
    plt.show()
    return anim
