"""
Synthetic datasets

@ Vivien Cabannes, 2023
"""
import numpy as np


# Dataset
def halfmoons(n, noise_level: float = 0.1, train: bool = True):
    """
    Return dataset in two dimensions populating two half-moons

    Parameters
    ----------
    n: int
        Number of sample
    noise_level
        Noise in the dataset, e.g. a value of .3 means overlap between clusters
    train
        If True, returns random train dataset; otherwise, some random parameter are linearly spaced

    Return
    ------
    out: ndarray of size (n, 2)
        Dataset
    """
    if train:
        theta = np.random.rand(n)
    else:
        theta = np.linspace(0, 1, n)
    theta *= 2 * np.pi
    out = np.empty((n, 2), dtype=float)
    out[:, 0] = np.cos(theta)
    out[:, 1] = np.sin(theta)
    out[out[:, 0] > 0, 1] += 1
    out += noise_level * np.random.randn(n, 2)
    return out


def concentric_circle(n, noise_level: float = 0.1, nb_circles: int = 4, train: bool = True):
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
    if train:
        theta = np.random.rand(n)
    else:
        theta = np.linspace(0, 1, n)
    theta *= 2 * np.pi
    out = np.empty((n, 2), dtype=float)
    out[:, 0] = np.cos(theta)
    out[:, 1] = np.sin(theta)
    out *= np.random.choice(nb_circles, n)[:, np.newaxis] + 1
    out += noise_level * np.random.randn(n, 2)
    return out


def swissroll(n, noise_level: float = 0.05, verbose: bool = False, train: bool = True):
    """
    Return dataset in three dimensions populating a swiss roll

    Parameters
    ----------
    n: int
        Number of sample
    noise_level
        Noise in the dataset
    verbose
        If True, returns the natural parameterization of the swiss roll
    train
        If True, returns random train dataset; otherwise, some random parameter are linearly spaced

    Return
    ------
    out: ndarray of size (n, 3)
        Dataset
    """
    if train:
        t = np.random.rand(n)
        y = np.random.rand(n)
    else:
        t = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        t, y = np.meshgrid(t, y)
    t *= 2
    t += 1
    t *= 3 * np.pi / 2
    out = np.zeros((*t.shape, 3), dtype=float)
    out[..., 0] = np.cos(t)
    out[..., 2] = np.sin(t)
    out *= t[..., np.newaxis]
    y *= 20
    out[..., 1] = y
    out += noise_level * np.random.randn(*out.shape)
    if verbose:
        return out, t
    return out
