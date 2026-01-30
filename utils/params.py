from dataclasses import dataclass
import numpy as np

@dataclass
class Params:
    """
    Parameters for the FFT2DSolver class.

    Args:
        f (callable): Initial vorticity field.
        Nx (int): Number of grid points in x direction.
        Ny (int): Number of grid points in y direction.
        Lx (float): Length of domain in x direction.
        Ly (float): Length of domain in y direction.
        dt (float): Time step.
        nt (int): Number of time steps.
        nt_save (int): Number of time steps to save.
        visc1 (np.ndarray): Viscosity term for first step.
        visc2 (np.ndarray): Viscosity term for second step.
    """
    f: callable
    Nx: int
    Ny: int
    Lx: float
    Ly: float
    dt: float
    nt: int
    nt_save: int
    visc1: np.ndarray = None
    visc2: np.ndarray = None

    def __post_init__(self):
            
        # Create discrete grid
        self.x = np.linspace(-0.5*self.Lx, 0.5*self.Lx, self.Nx, endpoint=False)
        self.y = np.linspace(-0.5*self.Ly, 0.5*self.Ly, self.Ny, endpoint=False)
        self.x_grid, self.y_grid = np.meshgrid(self.x, self.y)
        self.f = self.f(self.x_grid, self.y_grid)
        self.dx = self.Lx/self.Nx
        self.dy = self.Ly/self.Ny

        # Initialize wavenumbers
        self.kx = np.fft.fftfreq(self.Nx, self.Lx/self.Nx) * (2*np.pi)
        self.ky = np.fft.fftfreq(self.Ny, self.Ly/self.Ny) * (2*np.pi)
        self.kx_grid, self.ky_grid = np.meshgrid(self.kx, self.ky)
        self.k2 = self.kx_grid**2 + self.ky_grid**2
        self.k2[0,0] = 1.0e16 # Avoid division by zero
        

        # Initalize arrays
        self.wzq = np.zeros((self.Ny, self.Nx, int(np.ceil(self.nt/self.nt_save) + 1)))
        self.cfl = np.zeros((int(self.nt), 1))
        self.vx = np.zeros((self.Ny, self.Nx, int(np.ceil(self.nt/self.nt_save) + 1)))
        self.vy = np.zeros((self.Ny, self.Nx, int(np.ceil(self.nt/self.nt_save) + 1)))
        self.psi = np.zeros((self.Ny, self.Nx, int(np.ceil(self.nt/self.nt_save) + 1)))

        self.wzq[:, :, 0] = self.f

        # Initialize viscosity
        if self.visc1 is None:
            self.visc1 = np.exp(-((self.k2*(self.dx/np.pi)**2)**6)/16.0)
        if self.visc2 is None:
            self.visc2 = (1.0+self.visc1)/2.0