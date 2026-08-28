import numpy as np

import utils.params as params
from solver.fft2d import fft2d


class FFT2DSolver:
    def __init__(self, params: params.Params, **kwargs):
        self.params = params
        self.fft2d = lambda f, inverse: fft2d(
            f,
            self.params.Nx,
            self.params.Ny,
            inverse=inverse,
            dealias_mask=self.params.dealias_mask,
        )

    def _compute_derivative(self, vec):
        """
        Compute the derivative of a vector field in physical space using the Fourier transform
        of the vector field in spectral space.

        Args:
            vec (np.ndarray): 2D array representing the vector field.

        Returns:
            dvecdx_phys (np.ndarray): Derivative of vec along the x-axis in physical space.
            dvecdy_phys (np.ndarray): Derivative of vec along the y-axis in physical space.
        """
        dvecdx = 1j * self.params.kx_grid * vec  # i*k_x*vec
        dvecdy = 1j * self.params.ky_grid * vec  # i*k_y*vec
        dvecdx_phys = self.fft2d(
            dvecdx, inverse=True
        )  # Transform back to physical space
        dvecdy_phys = self.fft2d(dvecdy, inverse=True)
        return dvecdx_phys, dvecdy_phys

    def _compute_b_grad_j(self):
        """
        Compute (B * ∇)J term for the nonlinear advection term in the vorticity equation.

        Returns:
            nonlin (np.ndarray): Nonlinear advection term for the vorticity equation in spectral space.
            bx (np.ndarray): Magnetic field component along the x-axis in physical space.
            by (np.ndarray): Magnetic field component along the y-axis in physical space.
            Aphys (np.ndarray): Current density in physical space.
            djzdx_phys (np.ndarray): Derivative of jz along the x-axis in physical space.
            djzdy_phys (np.ndarray): Derivative of jz along the y-axis in physical space.
        """
        djzdx_phys, djzdy_phys = self._compute_derivative(self.jz)

        Ahat = self.jz / self.params.k2_div

        Aphys = self.fft2d(Ahat, inverse=True)

        # Computes magnetic field from curl(A)
        bx = self.fft2d(1j * self.params.ky_grid * Ahat, inverse=True) + self.params.b0x
        by = (
            self.fft2d(-1j * self.params.kx_grid * Ahat, inverse=True) + self.params.b0y
        )

        # Computes the one of the nonlinear advection terms using pseudospectral method
        nonlin = self.fft2d(bx * djzdx_phys + by * djzdy_phys, inverse=False)

        results = {
            "nonlin": nonlin,
            "bx": bx,
            "by": by,
            "A": Aphys,
            "djzdx_phys": djzdx_phys,
            "djzdy_phys": djzdy_phys,
        }

        return results

    def _compute_nonlinear_vorticity(self):
        """
        Compute the full nonlinear advection term for the vorticity equation
        using the pseudospectral method.

        Returns:
            nonlin (np.ndarray): Nonlinear advection term for the vorticity equation in spectral space.
            vx (np.ndarray): Advection velocity component along the x-axis in physical space.
            vy (np.ndarray): Advection velocity component along the y-axis in physical space.
            dwzdx_phs (np.ndarray): Derivative of wz along the x-axis in physical space.
            dwzdy_phs (np.ndarray): Derivative of wz along the y-axis in physical space.
        """
        # Compute nonlinear advection term for first step
        dwzdx_phys, dwzdy_phys = self._compute_derivative(self.wz)
        psi = self.wz / self.params.k2_div  # Compute stream function

        # Compute stream function in physical space
        psi_phys = self.fft2d(psi, inverse=True)

        # curl(psi) = v = (dpsi/dy, -dpsi/dx) = (i*ky*psi, -i*kx*psi) = (vx, vy)
        vx = self.fft2d(1j * self.params.ky_grid * psi, inverse=True)
        vy = self.fft2d(-1j * self.params.kx_grid * psi, inverse=True)

        # Get nonlinear advection term using advection equation
        nonlin = self.fft2d(-vx * dwzdx_phys - vy * dwzdy_phys, inverse=False)
        nonlin_current = self._compute_b_grad_j()
        nonlin_total = nonlin + nonlin_current["nonlin"]

        results = {
            "nonlin": nonlin_total,
            "vx": vx,
            "vy": vy,
            "psi": psi_phys,
            "dwzdx_phys": dwzdx_phys,
            "dwzdy_phys": dwzdy_phys,
            "bx": nonlin_current["bx"],
            "by": nonlin_current["by"],
            "A": nonlin_current["A"],
            "djzdx_phys": nonlin_current["djzdx_phys"],
            "djzdy_phys": nonlin_current["djzdy_phys"],
        }

        return results

    def _compute_nonlinear_current_density(
        self, vx, vy, bx, by, djzdx_phys, djzdy_phys, dwzdx_phys, dwzdy_phys, **kwargs
    ):
        """
        Compute the nonlinear advection term for the current density
        using the pseudospectral method.

        Args:
            vx (np.ndarray): Advection velocity component along the x-axis in physical space.
            vy (np.ndarray): Advection velocity component along the y-axis in physical space.
            bx (np.ndarray): Magnetic field component along the x-axis in physical space.
            by (np.ndarray): Magnetic field component along the y-axis in physical space.
            djzdx_phys (np.ndarray): Derivative of jz along the x-axis in physical space.
            djzdy_phys (np.ndarray): Derivative of jz along the y-axis in physical space.
            dwzdx_phys (np.ndarray): Derivative of wz along the x-axis in physical space.
            dwzdy_phys (np.ndarray): Derivative of wz along the y-axis in physical space.

        Returns:
            nonlin (np.ndarray): Nonlinear advection term for the current density in spectral space.
        """
        vdotgradj = self.fft2d(-vx * djzdx_phys - vy * djzdy_phys, inverse=False)
        bdotgradw = self.fft2d(bx * dwzdx_phys + by * dwzdy_phys, inverse=False)

        bxhat = self.fft2d(bx, inverse=False)
        byhat = self.fft2d(by, inverse=False)
        dbxdx_hat = 1j * self.params.kx_grid * bxhat
        dbxdy_hat = 1j * self.params.ky_grid * bxhat
        dbydx_hat = 1j * self.params.kx_grid * byhat
        dbxdx_phys = self.fft2d(dbxdx_hat, inverse=True)
        dbxdy_phys = self.fft2d(dbxdy_hat, inverse=True)
        dbydx_phys = self.fft2d(dbydx_hat, inverse=True)

        vxhat = self.fft2d(vx, inverse=False)
        vyhat = self.fft2d(vy, inverse=False)
        dvxdx_hat = 1j * self.params.kx_grid * vxhat
        dvxdy_hat = 1j * self.params.ky_grid * vxhat
        dvydx_hat = 1j * self.params.kx_grid * vyhat
        dvxdx_phys = self.fft2d(dvxdx_hat, inverse=True)
        dvxdy_phys = self.fft2d(dvxdy_hat, inverse=True)
        dvydx_phys = self.fft2d(dvydx_hat, inverse=True)

        almost_curl = 2 * dbxdx_phys * (dvxdy_phys + dvydx_phys) - 2 * dvxdx_phys * (
            dbxdy_phys + dbydx_phys
        )
        almost_curl_hat = self.fft2d(almost_curl, inverse=False)

        nonlin = vdotgradj + bdotgradw + almost_curl_hat

        results = {"nonlin": nonlin}

        return results

    def solve(self):

        # Initialize vorticity and transform to frequency space
        self.wz0 = self.fft2d(self.params.wzq[:, :, 0], inverse=False)
        self.wz = self.wz0.copy()

        self.jz0 = self.fft2d(self.params.jzq[:, :, 0], inverse=False)
        self.jz = self.jz0.copy()

        # Compute nonlinear advection term
        vorticity_results = self._compute_nonlinear_vorticity()
        current_density_results = self._compute_nonlinear_current_density(
            **vorticity_results
        )

        # Advect vorticity forward one step (Euler)
        vorticity_nonlin_old = vorticity_results["nonlin"]
        vx = vorticity_results["vx"]
        vy = vorticity_results["vy"]
        self.wz = (
            self.wz0 * self.params.visc1
            + self.params.dt * vorticity_nonlin_old * self.params.visc2
        )

        # Advect current density forward one step (Euler)
        cd_nonlin_old = current_density_results["nonlin"]
        self.jz = (
            self.jz0 * self.params.res1
            + self.params.dt * cd_nonlin_old * self.params.res2
        )
        self.params.cfl[0] = np.max(
            np.max(abs(np.sqrt(vx**2 + vy**2))) * self.params.dt / self.params.dx
        )
        print(self.params.cfl[0])

        # Save out results for first time step
        self.params.vx[:, :, 0] = vorticity_results["vx"]
        self.params.vy[:, :, 0] = vorticity_results["vy"]
        self.params.psi[:, :, 0] = vorticity_results["psi"]
        self.params.wzq[:, :, 0] = self.fft2d(self.wz0, inverse=True)
        self.params.jzq[:, :, 0] = self.fft2d(self.jz0, inverse=True)
        self.params.bx[:, :, 0] = vorticity_results["bx"]
        self.params.by[:, :, 0] = vorticity_results["by"]
        self.params.A[:, :, 0] = vorticity_results["A"]

        # Compute nonlinear advection term for corrector step
        vorticity_results = self._compute_nonlinear_vorticity()
        current_density_results = self._compute_nonlinear_current_density(
            **vorticity_results
        )
        vorticity_nonlin = vorticity_results["nonlin"]
        cd_nonlin = current_density_results["nonlin"]

        # Apply corrector step (trapezoidal rule)
        self.wz = (
            self.wz0 * self.params.visc1
            + 0.5
            * self.params.dt
            * (vorticity_nonlin + vorticity_nonlin_old)
            * self.params.visc2
        )
        self.jz = (
            self.jz0 * self.params.res1
            + 0.5 * self.params.dt * (cd_nonlin + cd_nonlin_old) * self.params.res2
        )

        for t in range(1, self.params.nt):
            vorticity_results = self._compute_nonlinear_vorticity()
            vorticity_nonlin = vorticity_results["nonlin"]
            vx = vorticity_results["vx"]
            vy = vorticity_results["vy"]
            psi_phys = vorticity_results["psi"]

            current_density_results = self._compute_nonlinear_current_density(
                **vorticity_results
            )
            cd_nonlin = current_density_results["nonlin"]

            self.params.cfl[t] = np.max(
                np.max(abs(np.sqrt(vx**2 + vy**2))) * self.params.dt / self.params.dx
            )

            if t % self.params.nt_save == 0:
                # Save out results for current time step every nt_save time steps
                self.params.vx[:, :, int(t / self.params.nt_save)] = vx.astype(
                    np.float64
                )
                self.params.vy[:, :, int(t / self.params.nt_save)] = vy.astype(
                    np.float64
                )
                self.params.psi[:, :, int(t / self.params.nt_save)] = psi_phys
                self.params.wzq[:, :, int(t / self.params.nt_save)] = self.fft2d(
                    self.wz, inverse=True
                )
                self.params.jzq[:, :, int(t / self.params.nt_save)] = self.fft2d(
                    self.jz, inverse=True
                )
                self.params.bx[:, :, int(t / self.params.nt_save)] = vorticity_results[
                    "bx"
                ].astype(np.float64)
                self.params.by[:, :, int(t / self.params.nt_save)] = vorticity_results[
                    "by"
                ].astype(np.float64)
                self.params.A[:, :, int(t / self.params.nt_save)] = vorticity_results[
                    "A"
                ]
                print(f"CFL: {self.params.cfl[t]}")

            # Advect forward one step (AB2 step)
            self.wz = (
                self.wz * self.params.visc1
                + self.params.dt
                * (1.5 * vorticity_nonlin - 0.5 * vorticity_nonlin_old)
                * self.params.visc2
            )
            self.jz = (
                self.jz * self.params.res1
                + self.params.dt
                * (1.5 * cd_nonlin - 0.5 * cd_nonlin_old)
                * self.params.res2
            )

            vorticity_nonlin_old = vorticity_nonlin
            cd_nonlin_old = cd_nonlin

        results = {
            "vorticity": self.params.wzq,
            "cfl": self.params.cfl,
            "vx": self.params.vx,
            "vy": self.params.vy,
            "psi": self.params.psi,
            "jz": self.params.jzq,
            "bx": self.params.bx,
            "by": self.params.by,
            "A": self.params.A,
            "k2": self.params.k2,
            "kx": self.params.kx,
            "ky": self.params.ky,
            "res1": self.params.res1,
            "res2": self.params.res2,
            "visc1": self.params.visc1,
            "visc2": self.params.visc2,
        }

        return results
