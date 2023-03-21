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
from .kernel_laplacian import KernelLaplacian


class GaussianKernel(KernelLaplacian):
    """
    Gaussian kernel.

    The Gaussian kernel is defined as
    .. math::
        K[i,j] = \exp(-\| (x1[i, :] - x2[j,:]) / \sigma \|^2)

    Parameters
    ----------
    sigma:
        Bandwidth parameter for the kernel

    Notes
    -----
    Class to build `L` with less flops than generic distance kernel computation.
    """

    def __init__(self, sigma: float = 1):
        super().__init__()
        self.sigma = sigma
        self.kernel_type = "gaussian_fast"

    def kernel(self, x1, x2, scap=None):
        r"""
        Computation of the Gaussian kernel
        .. math::
            K[i,j] = \exp(-\| x1[i, :] - x2[j,:] / \sigma \|^2)

        Parameters
        ----------
        x1: ndarray of size (n, d)
        x2: ndarray of size (n, d)
        scap: ndarray of size (n, n) (optional, default is None)
            Pre-computation of the scalar product `x1 @ x2.T`

        Returns
        -------
        K: ndarray of size (n, n)
        """
        K = distance_square(x1, x2, scap=scap)
        K /= -(self.sigma**2)
        np.exp(K, out=K)
        return K

    def laplacian(self, x_repr, x, K=None, X=None, X_repr=None, D=None):
        r"""
        Computation of the discrete Laplacian operator for the gaussian kernel
        .. math::
            L[i,j] = \sum_{k} \nabla_x k(x_k, y_i)^\top \nabla_x k(x_k, y_j)

        Parameters
        ----------
        x_repr: ndarray of size (p, d)
            ::math:`x_repr = y` representer to discretize `L^2`
        x: ndarray of size (n, d)
            Data to estimate the distribution on `L^2(X)`
        K: ndarray of size (p, n) (optional, default is None)
            Pre-computation of the kernel matrix `kernel(x_repr, x)`
        X: ndarray of size (p, n) (optional, default is None)
            Pre-computation of the dot-product matrix `x_repr.T @ x`
        X_repr: ndarray of size (p, p) (optional, default is None)
            Pre-computation of the dot-product matrix `x_repr.T @ x_repr`
        D: ndarray of size (n, ) (optional, default is None)
            Pre-computation of the square-norm vector `np.sum(x**2, axis=1)`

        Returns
        -------
        L: ndarray of size (p, p)
        """
        if K is None:
            K = self.kernel(x_repr, x)
        if X_repr is None:
            X_repr = scalar_product(x_repr, x_repr)
        if X is None:
            X = scalar_product(x_repr, x)
        if D is None:
            D = np.sum(x**2, axis=1)
        L = self._laplacian(K, X, X_repr, D)
        L *= 4 / self.sigma**4
        return L

    @staticmethod
    @numba.jit("f8[:, :](f8[:, :], f8[:, :], f8[:, :], f8[:])")
    def _laplacian(K, X, X_repr, D):
        p, n = K.shape
        out = np.zeros((p, p), dtype=np.float64)
        for i in range(p):
            for j in range(p):
                for k in range(n):
                    tmp = D[k] - X[i, k] - X[j, k] + X_repr[i, j]
                    out[i, j] += K[i, k] * K[j, k] * tmp
        return out
