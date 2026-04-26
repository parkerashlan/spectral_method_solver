import numpy as np


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
        g = np.fft.fft2(np.real(f))
        g[:, Nx // 2] = 0.0
        g[Ny // 2, :] = 0.0
    else:
        f = f.copy()
        f[:, Nx // 2] = 0.0
        f[Ny // 2, :] = 0.0
        g = np.real(np.fft.ifft2(f))
    return g
