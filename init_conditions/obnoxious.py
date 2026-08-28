import numpy as np

from functions.gaussian import gaussian2d


def init_functions():

    w0 = lambda x, y: (
        gaussian2d(x, y, 40, -0.25, -0.25, 1 / 64, 1 / 64) * np.sin(np.pi * x)
    )
    j0 = lambda x, y: (
        gaussian2d(x, y, 40, 0.25, 0.25, 1 / 64, 1 / 64) * np.cos(np.pi * x)
    )

    return w0, j0
