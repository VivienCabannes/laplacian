"""
Synthetic datasets

@ Vivien Cabannes, 2023
"""
import numpy as np


# Dataset
def halfmoons(n, noise_level: float = .1):
    """
    Return dataset in two dimensions populating two half-moons

    Parameters
    ----------
    n: int
        Number of sample
    noise_level
        Noise in the dataset, e.g. a value of .3 means overlap between clusters

    Return
    ------
    out: ndarray of size (n, 2)
        Dataset
    """
    theta = np.random.rand(n)
    theta *= 2 * np.pi
    out = np.empty((n, 2), dtype=float)
    out[:, 0] = np.cos(theta)
    out[:, 1] = np.sin(theta)
    out[out[:, 0] > 0, 1] += 1
    out += noise_level * np.random.randn(n, 2)
    return out


def concentric_circle(n, noise_level: float = .1, nb_circles: int = 4):
    """
    Return dataset in two dimensions populating concentric circles

    Parameters
    ----------
    n: int
        Number of sample
    noise_level
        Noise in the dataset, e.g. a value of .3 means overlap between clusters
    nb_circles
        Number of concentric circles in the dataset

    Return
    ------
    out: ndarray of size (n, 2)
        Dataset
    """
    theta = np.random.rand(n)
    theta *= 2 * np.pi
    out = np.empty((n, 2), dtype=float)
    out[:, 0] = np.cos(theta)
    out[:, 1] = np.sin(theta)
    out *= np.random.choice(nb_circles, n)[:, np.newaxis] + 1
    out += noise_level * np.random.randn(n, 2)
    return out
