import numpy as np

from solver.fft2d import fft2d
from solver.io_hdf5 import RunWriter
from utils import params


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

    def _cfl(self, vx, vy, bx, by):
        cfl = self.params.dt * np.max(
            (np.abs(vx) + np.abs(bx)) / self.params.dx
            + (np.abs(vy) + np.abs(by)) / self.params.dy
        )
        return cfl

    def _diagnostics(self, r, t_phys):
        dV = self.params.dx * self.params.dy
        return dict(
            t=t_phys,
            cfl=self._cfl(r["vx"], r["vy"], r["bx"], r["by"]),
            e_kin=0.5 * np.sum(r["vx"] ** 2 + r["vy"] ** 2) * dV,
            e_mag=0.5 * np.sum(r["bx"] ** 2 + r["by"] ** 2) * dV,
            jmax=np.abs(self.fft2d(self.jz, inverse=True)).max(),
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
        vx = self.fft2d(1j * self.params.ky_grid * psi, inverse=True) + self.params.v0x
        vy = self.fft2d(-1j * self.params.kx_grid * psi, inverse=True) + self.params.v0y

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

    def solve(self, outfile="run.h5"):

        with RunWriter(outfile, self.params) as out:
            # Initialize vorticity and transform to frequency space
            self.wz0 = self.fft2d(self.params.w, inverse=False)
            self.wz = self.wz0.copy()

            self.jz0 = self.fft2d(self.params.j, inverse=False)
            self.jz = self.jz0.copy()

            # Compute nonlinear advection term
            vorticity_results = self._compute_nonlinear_vorticity()
            current_density_results = self._compute_nonlinear_current_density(
                **vorticity_results
            )
            # Save out results for first time step
            out.write_scalars(
                0,
                **self._diagnostics(vorticity_results, 0),
            )
            out.append_fields(0.0, vorticity_results["A"], vorticity_results["psi"])

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
            print(self._cfl(vx, vy, vorticity_results["bx"], vorticity_results["by"]))

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

                out.write_scalars(
                    t,
                    **self._diagnostics(vorticity_results, t * self.params.dt),
                )

                current_density_results = self._compute_nonlinear_current_density(
                    **vorticity_results
                )
                cd_nonlin = current_density_results["nonlin"]

                if t % self.params.nt_save == 0:
                    out.append_fields(
                        t * self.params.dt,
                        vorticity_results["A"],
                        vorticity_results["psi"],
                    )
                    print(
                        f"CFL: {self._cfl(vorticity_results['vx'], vorticity_results['vy'], vorticity_results['bx'], vorticity_results['by'])}"
                    )

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

        results = {"outfile": str(outfile)}
        return results
