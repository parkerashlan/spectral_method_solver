import numpy as np

def gaussian2d(x, y, amplitude, center_x, center_y, sigma_x, sigma_y):
    gaussian = amplitude * np.exp(-((x - center_x) ** 2 / (2 * sigma_x ** 2) +
                                    (y - center_y) ** 2 / (2 * sigma_y ** 2)))
    return gaussian
