import numpy as np


def init_functions():

    psi_0 = 2
    w0 = lambda x, y: psi_0 * (-np.sin(y) + np.cos(x))

    A_0 = 1
    j0 = lambda x, y: A_0 * (-4 * np.cos(2 * y) + 2 * np.cos(x))

    return w0, j0
