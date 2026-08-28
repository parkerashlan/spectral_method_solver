import numpy as np


def init_functions():

    w0 = lambda x, y: 2 * np.pi * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))

    j0 = lambda x, y: (
        (1 / np.sqrt(4 * np.pi))
        * 2
        * np.pi
        * (2 * np.pi * np.cos(4 * np.pi * x) + np.cos(2 * np.pi * y))
    )

    return w0, j0
