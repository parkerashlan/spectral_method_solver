from collections import deque

from integrators.ab2_etd_approx import AB2ETDApprox


class AB3ETDApprox(AB2ETDApprox):
    """
    Third-order Adams-Bashforth stepper, same interface as AB2ETDApprox.

    x_new = x * c1 + dt * c2 * (23/12 f_n - 16/12 f_{n-1} + 5/12 f_{n-2})

    AB3 needs two previous f values, so startup is:
        step 1: Euler predictor + trapezoidal corrector (as in AB2)
        step 2: AB2
        step 3 onward: AB3
    """

    def __init__(self, coeffs, dt) -> None:
        super().__init__(coeffs, dt)
        self.f_hist = deque(maxlen=2)  # newest first: [f_{n-1}, f_{n-2}]

    def step(self, x, f, rhs):
        n_hist = len(self.f_hist)
        if n_hist == 0:
            x_new = self._startup(x, f, rhs)
        elif n_hist == 1:
            x_new = self._ab2(x, f, self.f_hist[0])
        else:
            x_new = self._ab3(x, f)

        self.f_hist.appendleft(f)
        return x_new

    def _ab3(self, x, f):
        f_nm1, f_nm2 = self.f_hist
        return tuple(
            xi * c1
            + self.dt
            * ((23.0 / 12.0) * fi - (16.0 / 12.0) * fnm1 + (5.0 / 12.0) * fnm2)
            * c2
            for xi, fi, fnm1, fnm2, (c1, c2) in zip(x, f, f_nm1, f_nm2, self.coeffs)
        )
