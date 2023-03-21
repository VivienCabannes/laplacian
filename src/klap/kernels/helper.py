"""
Linear algebra utilities

@ Vivien Cabannes, 2023
"""
import numpy as np


def scalar_product(x1, x2):
    r"""
    Computation of
    .. math::
        out[i,j] = \sum_{k} x1[i, k] * x2[j,k]

    Parameters
    ----------
    x1: ndarray of size (n, d)
    x2: ndarray of size (n, d)

    Returns
    -------
    out: ndarray of size (n, n)
    """
    return x1 @ x2.T


def distance_square(x1, x2, scap=None):
    r"""
    Computation of
    .. math::
        out[i,j] = \| x1[i, :] - x2[j, :] \|^2

    Parameters
    ----------
    x1: ndarray of size (n, d)
    x2: ndarray of size (n, d)
    scap: ndarray of size (n, n) (optional, default is None)
        Pre-computation of the scalar product `x1 @ x2.T`

    Returns
    -------
    out: ndarray of size (n, n)
    """
    if scap is None:
        out = scalar_product(x1, x2)
    else:
        out = scap.copy()
    out *= -2
    out += np.sum(x1**2, axis=1)[:, np.newaxis]
    out += np.sum(x2**2, axis=1)
    out[out < 0] = 0
    return out


def distance(x1, x2, scap=None):
    r"""
    Computation of
    .. math::
        out[i,j] = \| x1[i, :] - x2[j, :] \|

    Parameters
    ----------
    x1: ndarray of size (n, d)
    x2: ndarray of size (n, d)
    scap: ndarray of size (n, n) (optional, default is None)
        Pre-computation of the scalar product `x1 @ x2.T`

    Returns
    -------
    out: ndarray of size (n, n)
    """
    out = distance_square(x1, x2, scap=scap)
    np.sqrt(out, out=out)
    return out
