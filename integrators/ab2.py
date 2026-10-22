import numpy as np

from integrators.etd_coeffs import phi_functions


class AB2:
    """
    Second-order exponential time differencing (ETD2) stepper for a set of fields
    advanced together.

    Each field i has a linear operator L_i and is passed z_i = L_i * dt in `coeffs`.
    With E = e^z and the phi functions of z:

        x_new = E * x + dt * (weights applied to the nonlinear terms f)

    The first call to `step` bootstraps with an ETD1 predictor followed by an
    ETD2RK corrector, since AB2 needs one previous f. Later calls use ETD2 with
    the stored previous f.

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
        self.phis = tuple(phi_functions(z) for z in self.z)
        self.f_old = None

    def step(self, x, f, rhs):
        """
        Advance the state by one step.

        Args:
            x: tuple of fields at the current time.
            f: tuple of nonlinear terms evaluated at x.
            rhs: callable, rhs(x) -> tuple of nonlinear terms. Only used
                during startup for the corrector evaluation.

        Returns:
            Tuple of fields at the next time.
        """
        if self.f_old is None:
            x_new = self._startup(x, f, rhs)
        else:
            x_new = self._etd2(x, f, self.f_old)

        self.f_old = f
        return self._apply_filters(x_new)

    def _apply_filters(self, x):
        if self.filters is None:
            return x
        return tuple(xi if F is None else xi * F for xi, F in zip(x, self.filters))

    def _etd1(self, x, f):
        return tuple(
            E * xi + self.dt * p1 * fi
            for xi, fi, E, (p1, _, _) in zip(x, f, self.E, self.phis)
        )

    def _etd2rk(self, x, f, f_pred):
        return tuple(
            E * xi + self.dt * ((p1 - p2) * fi + p2 * fp)
            for xi, fi, fp, E, (p1, p2, _) in zip(x, f, f_pred, self.E, self.phis)
        )

    def _etd2(self, x, f, f_old):
        return tuple(
            E * xi + self.dt * ((p1 + p2) * fi - p2 * foi)
            for xi, fi, foi, E, (p1, p2, _) in zip(x, f, f_old, self.E, self.phis)
        )

    def _startup(self, x, f, rhs):
        x_pred = self._etd1(x, f)
        f_pred = rhs(x_pred)
        return self._etd2rk(x, f, f_pred)
