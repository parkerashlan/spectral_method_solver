import numpy as np


def init_functions():
    """
    Loop advection test (Gardiner & Stone).

    The fluid is at rest apart from a uniform background velocity
    V0 * (cos(theta), sin(theta)), which is set in the config with v0x / v0y.
    The magnetic potential is a cone of radius R centred at the origin.
    """
    w0 = lambda x, y: 0 * x + 0 * y

    A_0 = 1e-3
    R = 0.3

    def A0(x, y):
        r = np.sqrt(x**2 + y**2)
        return A_0 * np.maximum(R - r, 0.0)

    return w0, A0
