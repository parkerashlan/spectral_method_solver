import matplotlib.pyplot as plt
import numpy as np


def gaussian2d(x, y, mu, sigma):
    return (1 / (2 * np.pi * sigma**2)) * np.exp(
        -((x - mu) ** 2) / (2 * sigma**2) - (y - mu) ** 2 / (2 * sigma**2)
    )


def fft2d(f, Nx, Ny, inverse: bool = False):
    """FFT2D of a 2D array f in the direction specified by inverse.
    If inverse is False: real physical space --> complex frequency space
    If inverse is True: complex frequency space --> real physical space

    Args:
        f (np.ndarray): 2D array to be Fourier transformed.
        inverse (bool, optional): If True, the inverse FFT is computed. Defaults to False.

    Returns:
        g (np.ndarray): Real part of the Fourier transformed array.
    """
    if not inverse:
        g = np.fft.fft2(np.real(f), norm="forward")
        g[:, Nx // 2] = 0.0
        g[Ny // 2, :] = 0.0
    else:
        f = f.copy()
        f[:, Nx // 2] = 0.0
        f[Ny // 2, :] = 0.0
        g = np.real(np.fft.ifft2(f, norm="forward"))
    return g


Lx, Ly = 2 * np.pi, 2 * np.pi
nx, ny = 256, 256

dx = Lx / float(nx)
dy = Ly / float(ny)
x = np.linspace(-Lx / 2, Lx / 2, nx, endpoint=False)
y = np.linspace(-Ly / 2, Ly / 2, ny, endpoint=False)

X, Y = np.meshgrid(x, y)
mu = 0
sigma = 0.3
g = gaussian2d(X, Y, mu, sigma)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, g, cmap="viridis")
plt.show()

dx = Lx / float(nx)
dy = Ly / float(ny)
integrate = np.sum(g) * dx * dy
print(integrate)

kx = np.fft.fftfreq(nx, Lx / nx) * (2 * np.pi)
ky = np.fft.fftfreq(ny, Ly / ny) * (2 * np.pi)

ghat = fft2d(g, nx, ny)
ginv = fft2d(ghat, nx, ny, inverse=True)
print(np.real(ghat[0, 0]) * Lx * Ly)
print(np.sum(ginv) * dx * dy)
breakpoint()
