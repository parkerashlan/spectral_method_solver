from math import factorial

import numpy as np


def phi_functions(z, small=0.1, n_terms=10):
    """
    Return (phi1(z), phi2(z), phi3(z)) elementwise for real z.

    phi1(z) = (e^z - 1)/z and phi_{k+1}(z) = (phi_k(z) - 1/k!)/z.
    For |z| < small a Taylor series is used instead, since the closed forms
    lose all significant digits to cancellation as z -> 0.
    """
    z = np.asarray(z, dtype=float)
    is_small = np.abs(z) < small
    zs = np.where(is_small, 1.0, z)  # placeholder values, overwritten below

    phi1 = np.expm1(zs) / zs
    phi2 = (phi1 - 1.0) / zs
    phi3 = (phi2 - 0.5) / zs

    z_small = z[is_small]
    for phi, n in ((phi1, 1), (phi2, 2), (phi3, 3)):
        phi[is_small] = sum(z_small**k / factorial(k + n) for k in range(n_terms))

    return phi1, phi2, phi3
