from collections import deque

from integrators.ab2 import AB2


class AB3(AB2):
    """
    Third-order exponential time differencing (ETD3) stepper, same interface as AB2.

    x_new = E * x + dt * (g0 * f_n + g1 * f_{n-1} + g2 * f_{n-2})

    with g0 = phi1 + 3/2 phi2 + phi3, g1 = -2 phi2 - 2 phi3, g2 = 1/2 phi2 + phi3.
    As z -> 0 these reduce to the Adams-Bashforth weights (23/12, -16/12, 5/12).

    AB3 needs two previous f values, so startup is:
        step 1: ETD1 predictor + ETD2RK corrector (as in AB2)
        step 2: ETD2 (as in AB2)
        step 3 onward: ETD3
    """

    def __init__(self, coeffs, dt, filters=None) -> None:
        super().__init__(coeffs, dt, filters)
        self.f_hist = deque(maxlen=2)  # newest first: [f_{n-1}, f_{n-2}]
        self.g = tuple(
            (p1 + 1.5 * p2 + p3, -2.0 * p2 - 2.0 * p3, 0.5 * p2 + p3)
            for p1, p2, p3 in self.phis
        )

    def step(self, x, f, rhs):
        n_hist = len(self.f_hist)
        if n_hist == 0:
            x_new = self._startup(x, f, rhs)
        elif n_hist == 1:
            x_new = self._etd2(x, f, self.f_hist[0])
        else:
            x_new = self._ab3(x, f)

        self.f_hist.appendleft(f)
        return self._apply_filters(x_new)

    def _ab3(self, x, f):
        f_nm1, f_nm2 = self.f_hist
        return tuple(
            E * xi + self.dt * (g0 * fi + g1 * fnm1 + g2 * fnm2)
            for xi, fi, fnm1, fnm2, E, (g0, g1, g2) in zip(
                x, f, f_nm1, f_nm2, self.E, self.g
            )
        )
