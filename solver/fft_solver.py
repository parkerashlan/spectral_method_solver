import numpy as np
import matplotlib.pyplot as plt
import utils.params as params

class FFT2DSolver:
    def __init__(self, params: params.Params):
        self.params = params

    def fft2d(self, f, inverse: bool = False):
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
            g[:,self.params.Nx//2] = 0.0
            g[self.params.Ny//2,:] = 0.0
        else:
            f = f.copy()
            f[:,self.params.Nx//2] = 0.0
            f[self.params.Ny//2,:] = 0.0
            g = np.real(np.fft.ifft2(f))
        return g

    def _compute_nonlinear_advection(self):
        # Compute nonlinear advection term for first step
        dwzdx = 1j*self.params.kx_grid*self.wz # i*k_x*wz
        dwzdy = 1j*self.params.ky_grid*self.wz # i*k_y*wz
        dwzdx_phys = self.fft2d(dwzdx, inverse=True) # Transform back to physical space
        dwzdy_phys = self.fft2d(dwzdy, inverse=True)
        psi = 1j*self.wz/self.params.k2 # Compute stream function

        # Compute stream function in physical space
        psi_phys = self.fft2d(psi, inverse=True)

        # -curl(psi) = v = (-dpsi/dy, dpsi/dx) = (-i*ky*psi, i*kx*psi) = (vx, vy)
        vx = self.fft2d(self.params.ky_grid*psi, inverse=True)
        vy = self.fft2d(-self.params.kx_grid*psi, inverse=True)
        
        # Get nonlinear advection term using advection equation
        nonlin = self.fft2d(-vx*dwzdx_phys-vy*dwzdy_phys, inverse=False)

        results = {'nonlin': nonlin, 
                   'vx': vx, 
                   'vy': vy, 
                   'psi': psi_phys}

        return results

    def solve(self):
        
        # Initialize vorticity and transform to frequency space
        self.wz0 = self.fft2d(self.params.wzq[:, :, 0], inverse=False)
        self.wz = self.wz0.copy()

        # Compute nonlinear advection term
        results = self._compute_nonlinear_advection()
        nonlin_old = results['nonlin']
        vx = results['vx']
        vy = results['vy']
        self.params.cfl[0] = np.max(np.max(abs(vx)))*self.params.dt/self.params.dx

        # Advect forward one step (Euler)
        self.wz = self.wz*self.params.visc1 + self.params.dt*nonlin_old*self.params.visc2

        # Compute nonlinear advection term for corrector step
        results = self._compute_nonlinear_advection()
        nonlin = results['nonlin']
        
        # Apply corrector step (trapezoidal rule)
        self.wz = self.wz*self.params.visc1 + 0.5*self.params.dt*(nonlin+nonlin_old)*self.params.visc2
        
        # Save out results for first time step
        self.params.vx[:, :, 0] = results['vx']
        self.params.vy[:, :, 0] = results['vy']
        self.params.psi[:, :, 0] = results['psi']
        self.params.wzq[:, :, 0] = self.fft2d(self.wz, inverse=True)
        
        for t in range(1, self.params.nt):
            results = self._compute_nonlinear_advection()
            nonlin = results['nonlin']
            vx = results['vx']
            vy = results['vy']
            psi_phys = results['psi']

            self.params.cfl[t] = np.max(np.max(abs(vx)))*self.params.dt/self.params.dx

            # Advect forward one step (AB2 step)
            self.wz = self.wz*self.params.visc1 + self.params.dt*(1.5*nonlin-0.5*nonlin_old)*self.params.visc2

            if t % self.params.nt_save == 0:
                # Save out results for current time step every nt_save time steps
                self.params.vx[:, :, int(t/self.params.nt_save)] = vx
                self.params.vy[:, :, int(t/self.params.nt_save)] = vy
                self.params.psi[:, :, int(t/self.params.nt_save)] = psi_phys
                self.params.wzq[:, :, int(t/self.params.nt_save)] = self.fft2d(self.wz, inverse=True)
            nonlin_old = nonlin

        results = {'vorticity': self.params.wzq, 
                   'cfl': self.params.cfl,
                   'vx': self.params.vx,
                   'vy': self.params.vy,
                   'psi': self.params.psi}

        return results
