import numpy as np

from integrators.etd_coeffs import phi_functions


class RK4:
    """
    Fourth-order exponential time differencing Runge-Kutta stepper (ETDRK4,
    Cox-Matthews) for a set of fields advanced together.

    Solves x' = L x + N(x) with L diagonal. Each field i is passed z_i = L_i * dt
    in `coeffs`. The linear part is integrated exactly, so stiff damping does
    not restrict dt. As z -> 0 this reduces to classical RK4.

    Args:
        coeffs: sequence of z arrays, one per field, e.g. (z_visc, z_res).
        dt: time step.
        filters: optional sequence with one entry per field, each an array or
            None. If given, the state is multiplied by the filter at the end of
            every step (used for hyperviscosity).
    """

    def __init__(self, coeffs, dt, filters=None) -> None:
        self.z = tuple(coeffs)
        self.dt = dt
        self.filters = filters
        self.E = tuple(np.exp(z) for z in self.z)
        self.E2 = tuple(np.exp(z / 2) for z in self.z)
        self.Q = tuple(0.5 * dt * phi_functions(z / 2)[0] for z in self.z)
        self.f_w = tuple(
            (
                dt * (p1 - 3.0 * p2 + 4.0 * p3),
                dt * (p2 - 2.0 * p3),
                dt * (-p2 + 4.0 * p3),
            )
            for p1, p2, p3 in (phi_functions(z) for z in self.z)
        )

    def step(self, x, f, rhs):
        """
        Advance the state by one step.

        Args:
            x: tuple of fields at the current time.
            f: tuple of nonlinear terms evaluated at x.
            rhs: callable, rhs(x) -> tuple of nonlinear terms. Called three times.

        Returns:
            Tuple of fields at the next time.
        """
        a = tuple(E2 * xi + Q * fi for xi, fi, E2, Q in zip(x, f, self.E2, self.Q))
        f_a = rhs(a)

        b = tuple(E2 * xi + Q * fai for xi, fai, E2, Q in zip(x, f_a, self.E2, self.Q))
        f_b = rhs(b)

        c = tuple(
            E2 * ai + Q * (2.0 * fbi - fi)
            for ai, fbi, fi, E2, Q in zip(a, f_b, f, self.E2, self.Q)
        )
        f_c = rhs(c)

        x_new = tuple(
            E * xi + f1 * fi + 2.0 * f2 * (fai + fbi) + f3 * fci
            for xi, fi, fai, fbi, fci, E, (f1, f2, f3) in zip(
                x, f, f_a, f_b, f_c, self.E, self.f_w
            )
        )
        return self._apply_filters(x_new)

    def _apply_filters(self, x):
        if self.filters is None:
            return x
        return tuple(xi if F is None else xi * F for xi, F in zip(x, self.filters))
