import numpy as np


def init_functions():
    v1 = 2.0
    k = 2
    w0 = lambda x, y: v1 * k * np.sin(k * x)

    j0 = lambda x, y: 0 * x + 0 * y

    return w0, j0
