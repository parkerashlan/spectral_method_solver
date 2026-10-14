import numpy as np


def init_functions():
    v1 = 2.0
    k = 2
    w0 = lambda x, y: v1 * k * np.sin(k * x)

    A0 = lambda x, y: -(v1 / k) * np.sin(k * x)

    return w0, A0
