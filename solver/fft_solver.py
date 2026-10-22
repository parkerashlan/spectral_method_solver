from functools import partial
from pathlib import Path

import numpy as np

from integrators.ab2 import AB2
from integrators.ab2_etd_approx import AB2ETDApprox
from integrators.ab3 import AB3
from integrators.ab3_etd_approx import AB3ETDApprox
from integrators.rk4 import RK4
from solver.fft2d import fft2d
from solver.io_hdf5 import RunWriter
from utils import params

INTEGRATORS = {
    "ab2": AB2,
    "ab3": AB3,
    "rk4": RK4,
    "ab2-etd-approx": AB2ETDApprox,
    "ab3-etd-approx": AB3ETDApprox,
}


class FFT2DSolver:
    def __init__(self, params: params.Params, **kwargs):
        self.params = params
        self.fft2d = partial(
            fft2d,
            Nx=self.params.Nx,
            Ny=self.params.Ny,
            dealias_mask=self.params.dealias_mask,
        )

    def _cfl(self, vx, vy, bx, by):
        cfl = self.params.dt * np.max(
            (np.abs(vx) + np.abs(bx)) / self.params.dx
            + (np.abs(vy) + np.abs(by)) / self.params.dy
        )
        return cfl

    def _diagnostics(self, r, t_phys, Az):
        dV = self.params.dx * self.params.dy
        return {
            "t": t_phys,
            "cfl": self._cfl(r["vx"], r["vy"], r["bx"], r["by"]),
            "e_kin": 0.5 * np.sum(r["vx"] ** 2 + r["vy"] ** 2) * dV,
            "e_mag": 0.5 * np.sum(r["bx"] ** 2 + r["by"] ** 2) * dV,
            "jmax": np.abs(self.fft2d(self.params.k2 * Az, inverse=True)).max(),
        }

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

    def _compute_b_grad_j(self, Az):
        """
        Compute (B * ∇)J term for the nonlinear advection term in the vorticity equation.

        Args:
            Az (np.ndarray): Vector potential in spectral space.

        Returns:
            nonlin (np.ndarray): Nonlinear advection term for the vorticity equation in spectral space.
            bx (np.ndarray): Magnetic field component along the x-axis in physical space.
            by (np.ndarray): Magnetic field component along the y-axis in physical space.
            Aphys (np.ndarray): Vector potential in physical space.
            djzdx_phys (np.ndarray): Derivative of jz along the x-axis in physical space.
            djzdy_phys (np.ndarray): Derivative of jz along the y-axis in physical space.
        """
        jz = self.params.k2 * Az
        djzdx_phys, djzdy_phys = self._compute_derivative(jz)

        Aphys = self.fft2d(Az, inverse=True)

        # Computes magnetic field from curl(A)
        bx = self.fft2d(1j * self.params.ky_grid * Az, inverse=True) + self.params.b0x
        by = self.fft2d(-1j * self.params.kx_grid * Az, inverse=True) + self.params.b0y

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

    def _compute_nonlinear_vorticity(self, wz, Az):
        """
        Compute the full nonlinear advection term for the vorticity equation
        using the pseudospectral method.

        Args:
            wz (np.ndarray): Vorticity in spectral space.
            Az (np.ndarray): Vector potential in spectral space.

        Returns:
            nonlin (np.ndarray): Nonlinear advection term for the vorticity equation in spectral space.
            vx (np.ndarray): Advection velocity component along the x-axis in physical space.
            vy (np.ndarray): Advection velocity component along the y-axis in physical space.

            psi (np.ndarray): Stream function in physical space.
            psi_hat (np.ndarray): Stream function in spectral space.
            dwzdx_phys (np.ndarray): Derivative of wz along the x-axis in physical space.
            dwzdy_phys (np.ndarray): Derivative of wz along the y-axis in physical space.
            bx (np.ndarray): Magnetic field component along the x-axis in physical space.
            by (np.ndarray): Magnetic field component along the y-axis in physical space.
            A (np.ndarray): Vector potential in physical space.
            djzdx_phys (np.ndarray): Derivative of jz along the x-axis in physical space.
            djzdy_phys (np.ndarray): Derivative of jz along the y-axis in physical space.
        """
        # Compute nonlinear advection term for first step
        dwzdx_phys, dwzdy_phys = self._compute_derivative(wz)
        psi = wz / self.params.k2_div  # Compute stream function

        # Compute stream function in physical space
        psi_phys = self.fft2d(psi, inverse=True)

        # curl(psi) = v = (dpsi/dy, -dpsi/dx) = (i*ky*psi, -i*kx*psi) = (vx, vy)
        vx = self.fft2d(1j * self.params.ky_grid * psi, inverse=True) + self.params.v0x
        vy = self.fft2d(-1j * self.params.kx_grid * psi, inverse=True) + self.params.v0y

        # Get nonlinear advection term using advection equation
        nonlin = self.fft2d(-vx * dwzdx_phys - vy * dwzdy_phys, inverse=False)
        nonlin_current = self._compute_b_grad_j(Az)
        nonlin_total = nonlin + nonlin_current["nonlin"]

        results = {
            "nonlin": nonlin_total,
            "vx": vx,
            "vy": vy,
            "psi": psi_phys,
            "psi_hat": psi,
            "dwzdx_phys": dwzdx_phys,
            "dwzdy_phys": dwzdy_phys,
            "bx": nonlin_current["bx"],
            "by": nonlin_current["by"],
            "A": nonlin_current["A"],
            "djzdx_phys": nonlin_current["djzdx_phys"],
            "djzdy_phys": nonlin_current["djzdy_phys"],
        }

        return results

    def _compute_nonlinear_vec_potential(self, bx, by, psi_hat, **kwargs):
        """
        Compute the nonlinear term for the vector potential equation,
        dA/dt = B . grad(psi) + eta * laplacian(A) (= -v . grad(A) + resistive term),
        using the pseudospectral method. Only the nonlinear part is computed here.

        Args:
            bx (np.ndarray): Magnetic field component along the x-axis in physical space (including b0x).
            by (np.ndarray): Magnetic field component along the y-axis in physical space (including b0y).
            psi_hat (np.ndarray): Stream function in spectral space.
            **kwargs: Unused extra entries from the vorticity results dictionary.

        Returns:
            results (dict): Dictionary with key "nonlin", the nonlinear term for the
                vector potential in spectral space.
        """
        dpsidx_phys, dpsidy_phys = self._compute_derivative(psi_hat)
        dpsidx_phys = dpsidx_phys - self.params.v0y
        dpsidy_phys = dpsidy_phys + self.params.v0x
        nonlin = self.fft2d(bx * dpsidx_phys + by * dpsidy_phys, inverse=False)

        results = {"nonlin": nonlin}
        return results

    def _rhs(self, state):
        """
        Evaluate the nonlinear terms for a given state.

        Depends only on `state`; nothing is stored on self.

        Args:
            state (tuple): (wz, Az) in spectral space.

        Returns:
            nonlin (tuple): (vorticity nonlinear term, vector potential nonlinear term).
            vorticity_results (dict): Physical-space fields for diagnostics and output.
        """
        wz, Az = state
        vorticity_results = self._compute_nonlinear_vorticity(wz, Az)
        vec_potential_results = self._compute_nonlinear_vec_potential(
            **vorticity_results
        )
        nonlin = (vorticity_results["nonlin"], vec_potential_results["nonlin"])
        return nonlin, vorticity_results

    def solve(self, outfile: str | Path = "run.h5"):

        with RunWriter(outfile, self.params) as out:
            # Initialize vorticity and transform to frequency space
            wz0 = self.fft2d(self.params.w, inverse=False)
            Az0 = self.fft2d(self.params.A, inverse=False)

            name = self.params.integrator
            if name.endswith("-etd-approx"):
                stepper = INTEGRATORS[name](
                    coeffs=(
                        (self.params.visc1, self.params.visc2),
                        (self.params.res1, self.params.res2),
                    ),
                    dt=self.params.dt,
                )
            else:
                stepper = INTEGRATORS[name](
                    coeffs=(self.params.z_visc, self.params.z_res),
                    dt=self.params.dt,
                    filters=(self.params.visc_filter, self.params.res_filter),
                )
            rhs = lambda state: self._rhs(state)[0]

            # Compute nonlinear advection term
            state = (wz0.copy(), Az0.copy())
            nonlin, vorticity_results = self._rhs(state)

            # Save out results for first time step
            out.write_scalars(
                0,
                **self._diagnostics(vorticity_results, 0, state[1]),
            )
            out.append_fields(0.0, vorticity_results["A"], vorticity_results["psi"])

            vx = vorticity_results["vx"]
            vy = vorticity_results["vy"]
            print(self._cfl(vx, vy, vorticity_results["bx"], vorticity_results["by"]))

            # First step (Euler predictor + trapezoidal corrector, handled by the stepper)
            state = stepper.step(state, nonlin, rhs)

            for t in range(1, self.params.nt):
                nonlin, vorticity_results = self._rhs(state)

                out.write_scalars(
                    t,
                    **self._diagnostics(
                        vorticity_results, t * self.params.dt, state[1]
                    ),
                )

                if t % self.params.nt_save == 0:
                    out.append_fields(
                        t * self.params.dt,
                        vorticity_results["A"],
                        vorticity_results["psi"],
                    )
                    print(
                        f"CFL: {self._cfl(vorticity_results['vx'], vorticity_results['vy'], vorticity_results['bx'], vorticity_results['by'])}"
                    )

                # Advect forward one step
                state = stepper.step(state, nonlin, rhs)

        results = {"outfile": str(outfile)}
        return results
