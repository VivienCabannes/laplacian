import numpy as np


def meshgrid_3d(n):
    theta = np.linspace(0, 2 * np.pi, n)
    phi = np.linspace(0, np.pi, n)
    X = np.outer(np.cos(theta), np.sin(phi))
    Y = np.outer(np.sin(theta), np.sin(phi))
    Z = np.outer(np.ones(n), np.cos(phi))
    return X, Y, Z
