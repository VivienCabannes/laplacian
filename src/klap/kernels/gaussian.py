"""
Fast implementation of operator at play to estimate Laplacian eigendecomposition.

@ Vivien Cabannes, 2023
"""
import numba
import numpy as np
from .helper import (
    distance_square,
    scalar_product,
)


def rbf_kernel(x1, x2, sigma: float = 1):
    r"""
    Computation of the Gaussian kernel
    .. math::
        K[i,j] = \exp(-\| x1[i, :] - x2[j,:] / \sigma \|^2)

    Parameters
    ----------
    x1: ndarray of size (n, d)
    x2: ndarray of size (n, d)
    sigma
        Bandwidth parameter for the kernel

    Returns
    -------
    K: ndarray of size (n, n)
    """
    K = distance_square(x1, x2)
    K /= -(sigma**2)
    np.exp(K, out=K)
    return K


def rbf_laplacian(x_repr, x, sigma: float = 1, K=None):
    r"""
    Computation of the discrete Laplacian operator for the gaussian kernel
    .. math::
        L[i,j] = \sum_{k} \nabla_{x} k(x_k, y_i)^\top \nabla_x k(x_i, y_j)

    Parameters
    ----------
    x_repr: ndarray of size (p, d)
        ::math:`x_repr = y` representer to discretize `L^2`
    x: ndarray of size (n, d)
        Data to estimate the distribution on `L^2(X)`
    sigma
        Bandwidth parameter for the kernel
    K: ndarray of size (p, n) (optional, default is None)
        Pre-computation of the gaussian kernel `k(x_repr, x)`

    Returns
    -------
    L: ndarray of size (p, p)
    """
    if K is None:
        K = rbf_kernel(x_repr, x, sigma=sigma)
    S_in = scalar_product(x_repr, x_repr)
    S = scalar_product(x_repr, x)
    S_out = np.sum(x**2, axis=1)
    L = _rbf_laplacian(K, S, S_in, S_out)
    L *= 4 / sigma**4
    return L


@numba.jit("f8[:, :](f8[:, :], f8[:, :], f8[:, :], f8[:])")
def _rbf_laplacian(K, S, S_in, S_out):
    p, n = K.shape
    out = np.zeros((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(p):
            for k in range(n):
                tmp = S_out[k] - S[i, k] - S[j, k] + S_in[i, j]
                out[i, j] += K[i, k] * K[j, k] * tmp
    return out
