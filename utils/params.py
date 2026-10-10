from dataclasses import dataclass

import numpy as np


@dataclass
class Params:
    """
    Parameters for the FFT2DSolver class.

    Args:
        w0 (callable): Initial vorticity field.
        j0 (callable): Initial current density.
        Nx (int): Number of grid points in x direction.
        Ny (int): Number of grid points in y direction.
        Lx (float): Length of domain in x direction.
        Ly (float): Length of domain in y direction.
        dt (float): Time step.
        nt (int): Number of time steps.
        nt_save (int): Number of time steps to save.
        nu (float): Viscosity coefficient.
        eta (float): Magnetic resistivity coefficient.
    """

    w0: callable
    j0: callable
    Nx: int
    Ny: int
    Lx: float
    Ly: float
    dt: float
    nt: int
    nt_save: int
    nu: float = 0.0001
    eta: float = 0.0001
    hyper: bool = False
    b0x: float = 0.0
    b0y: float = 0.0
    v0x: float = 0.0
    v0y: float = 0.0
    symm: bool = True

    def __post_init__(self):

        # Create discrete grid
        if self.symm:
            self.x = np.linspace(-0.5 * self.Lx, 0.5 * self.Lx, self.Nx, endpoint=False)
            self.y = np.linspace(-0.5 * self.Ly, 0.5 * self.Ly, self.Ny, endpoint=False)
        else:
            self.x = np.linspace(0, self.Lx, self.Nx, endpoint=False)
            self.y = np.linspace(0, self.Ly, self.Ny, endpoint=False)
        self.x_grid, self.y_grid = np.meshgrid(self.x, self.y)
        self.w = self.w0(self.x_grid, self.y_grid)
        self.j = self.j0(self.x_grid, self.y_grid)
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny

        # Initialize wavenumbers
        # Note: Scaled by 2*pi to convert to angular wavenumbers of rad/length instead of cycles/length
        # This also means we don't need to use 1j * 2pi * kx when doing derivatives
        # Note: kx uses rfftfreq (not fftfreq) because fft2d uses rfft2/irfft2
        # under the hood, which only computes the non-redundant half of the
        # spectrum along the last transformed axis (Nx). All spectral-space
        # arrays (and therefore kx_grid/ky_grid) must match that (Ny, Nx//2+1)
        # shape or broadcasting will fail against the dealias mask, etc.
        self.kx = np.fft.rfftfreq(self.Nx, self.Lx / self.Nx) * (2 * np.pi)
        self.ky = np.fft.fftfreq(self.Ny, self.Ly / self.Ny) * (2 * np.pi)
        self.kx_grid, self.ky_grid = np.meshgrid(self.kx, self.ky)
        self.k2 = self.kx_grid**2 + self.ky_grid**2
        self.k2_div = self.k2.copy()
        self.k2_div[0, 0] = 1.0e64  # Avoid division by zero
        # self.k2[0, 0] = 1.0e16  # Avoid division by zero

        # Implement 2/3 dealiasing rule
        self.kx_cut = (2 * np.pi / self.Lx) * (self.Nx // 3)
        self.ky_cut = (2 * np.pi / self.Ly) * (self.Ny // 3)
        self.dealias_mask = (np.abs(self.kx_grid) <= self.kx_cut) & (
            np.abs(self.ky_grid) <= self.ky_cut
        )

        # Initalize arrays
        # self.wzq = np.zeros(
        #     (self.Ny, self.Nx, int(np.ceil(self.nt / self.nt_save) + 1))
        # )
        # self.vx = np.zeros((self.Ny, self.Nx, int(np.ceil(self.nt / self.nt_save) + 1)))
        # self.vy = np.zeros((self.Ny, self.Nx, int(np.ceil(self.nt / self.nt_save) + 1)))
        # self.psi = np.zeros(
        #     (self.Ny, self.Nx, int(np.ceil(self.nt / self.nt_save) + 1))
        # )
        # # Initialize electromagnetic arrays
        # self.jzq = np.zeros(
        #     (self.Ny, self.Nx, int(np.ceil(self.nt / self.nt_save) + 1))
        # )
        # self.bx = np.zeros((self.Ny, self.Nx, int(np.ceil(self.nt / self.nt_save) + 1)))
        # self.by = np.zeros((self.Ny, self.Nx, int(np.ceil(self.nt / self.nt_save) + 1)))
        # self.A = np.zeros((self.Ny, self.Nx, int(np.ceil(self.nt / self.nt_save) + 1)))
        self.cfl = None

        # self.wzq[:, :, 0] = self.w
        # self.jzq[:, :, 0] = self.j

        # Initialize viscosity
        # Del^2 or set it to the power, add setting for hyperviscosity
        if not self.hyper:
            self.res1 = np.exp(-self.eta * self.k2 * self.dt)
            self.res2 = (1.0 - self.res1) / (self.eta * self.k2_div * self.dt)
            self.visc1 = np.exp(-self.nu * self.k2 * self.dt)
            self.visc2 = (1.0 - self.visc1) / (self.nu * self.k2_div * self.dt)
        else:
            self.visc1 = np.exp(-((self.k2 * (self.dx / np.pi) ** 2) ** 4))
            self.visc2 = (1.0 + self.visc1) / 2.0

            self.res1 = np.exp(-((self.k2 * (self.dx / np.pi) ** 2) ** 4))
            self.res2 = (1.0 + self.res1) / 2.0

    def __iter__(self):
        return iter(vars(self).items())
