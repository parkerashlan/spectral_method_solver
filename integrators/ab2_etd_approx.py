from typing import Any


class AB2ETDApprox:
    """
    Second-order Adams-Bashforth stepper for a set of fields advanced together.

    Each field i is advanced with its own pair of coefficients (c1_i, c2_i):

        x_new = x * c1 + dt * f * c2

    where f is the Euler / trapezoid / AB2 combination of nonlinear terms.
    The first call to `step` bootstraps with an Euler predictor followed by a
    trapezoidal (Heun) corrector, since AB2 needs one previous f. Later calls
    use the AB2 formula with the stored previous f.

    Args:
        coeffs: sequence of (c1, c2) pairs, one per field, e.g.
            ((visc1, visc2), (res1, res2)).
        dt: time step.
    """

    def __init__(self, coeffs, dt) -> None:
        self.coeffs = tuple(coeffs)
        self.dt = dt
        self.f_old: tuple[Any, ...] | None = None

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
            x_new = self._ab2(x, f, self.f_old)

        self.f_old = f
        return x_new

    def _euler(self, x, f):
        return tuple(
            xi * c1 + self.dt * fi * c2 for xi, fi, (c1, c2) in zip(x, f, self.coeffs)
        )

    def _heun(self, x, f, f_pred):
        return tuple(
            xi * c1 + 0.5 * self.dt * (fp + fi) * c2
            for xi, fi, fp, (c1, c2) in zip(x, f, f_pred, self.coeffs)
        )

    def _ab2(self, x, f, f_old):
        return tuple(
            xi * c1 + self.dt * (1.5 * fi - 0.5 * foi) * c2
            for xi, fi, foi, (c1, c2) in zip(x, f, f_old, self.coeffs)
        )

    def _startup(self, x, f, rhs):
        x_pred = self._euler(x, f)
        f_pred = rhs(x_pred)
        return self._heun(x, f, f_pred)
