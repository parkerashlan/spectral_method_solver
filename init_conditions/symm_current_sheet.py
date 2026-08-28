import numpy as np


def current_sheet(x, y, strength, sheet_loc, sign):
    """
    Creates a current sheet centered at `sheet_loc`.

    Parameters
    ----------
    x : array-like
        The x-coordinates.
    y : array-like
        The y-coordinates.
    strength : float
        The strength of the current sheet.
    sheet_loc : float
        The location of the sheet.
    sign : int
        The sign of the current sheet (1 or -1).

    Returns
    -------
    array-like
        The current sheet values at (x, y).
    """
    if sign not in [1, -1]:
        raise ValueError("sign must be 1 or -1")
    return strength * (sign * (1 / np.cosh(strength * (x + sheet_loc)) ** 2))


def init_functions():
    u0 = 0.001
    w0 = lambda x, y: -2 * np.pi * u0 * np.cos(2 * np.pi * y)
    j0 = lambda x, y: (
        current_sheet(x, y, strength=1.0, sheet_loc=0.5, sign=1)
        + current_sheet(x, y, strength=1.0, sheet_loc=-0.5, sign=-1)
    )
    return w0, j0
