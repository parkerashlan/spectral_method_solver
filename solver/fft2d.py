import numpy as np


def fft2d(
    f: np.ndarray,
    Nx: int,
    Ny: int,
    inverse: bool = False,
    dealias_mask: np.ndarray | None = None,
) -> np.ndarray:
    """FFT2D of a 2D or 3D array f in the direction specified by inverse.
    If inverse is False: real physical space --> complex frequency space
    If inverse is True: complex frequency space --> real physical space

    Args:
        f (np.ndarray): 2D array to be Fourier transformed.
        inverse (bool, optional): If True, the inverse FFT is computed. Defaults to False.
        Nx (int): Number of grid points in the x direction.
        Ny (int): Number of grid points in the y direction.
        dealias_mask (np.ndarray, optional): Mask to apply dealiasing. Defaults to None.

    Returns:
        g (np.ndarray): Real part of the Fourier transformed array.
    """
    if dealias_mask is None:
        # Physical-space arrays are shaped (Ny, Nx), and rfft2/irfft2 only
        # keep the non-redundant half of the spectrum along the last
        # transformed axis (Nx), so the spectral-space shape is
        # (Ny, Nx // 2 + 1), not the full (Nx, Ny).
        dealias_mask = np.ones((Ny, Nx // 2 + 1), dtype=np.float64)

    def _broadcastable_mask(mask, target_ndim):
        # The mask only describes the leading (Ny, Nx_half) spectral axes.
        # Callers may pass in arrays with extra trailing axes (e.g. a time
        # axis), so pad the mask with trailing singleton axes to keep it
        # broadcastable against those.
        if mask.ndim < target_ndim:
            mask = mask.reshape(mask.shape + (1,) * (target_ndim - mask.ndim))
        return mask

    # Note: Able to use rfft2/irfft2 since all of my fields are real-valued
    # This saves on memory and computation time compared to fft2/ifft2.
    if not inverse:
        f = np.asarray(f, dtype=np.float64)
        g = np.fft.rfft2(np.real(f), axes=(0, 1))  # , norm="forward")
        g *= _broadcastable_mask(dealias_mask, g.ndim)
    else:
        f = np.asarray(f, dtype=np.complex128)
        f = f.copy()
        f *= _broadcastable_mask(dealias_mask, f.ndim)
        g = np.real(np.fft.irfft2(f, s=(Ny, Nx), axes=(0, 1))).astype(
            np.float64
        )  # , norm="forward"))
    return g
