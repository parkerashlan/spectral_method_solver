import numpy as np


def init_functions():

    psi_0 = np.sqrt(2) / 2
    w0 = lambda x, y: -2 * psi_0 * (np.sin(x) * np.sin(y))

    A_0_const = 1
    A0 = lambda x, y: A_0_const * (-np.cos(x) + 1 / 2 * np.cos(2 * y))

    return w0, j0
