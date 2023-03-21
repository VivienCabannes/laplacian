"""
Fast implementation of operator at play to estimate Laplacian eigendecomposition with distance kernel.

@ Vivien Cabannes, 2023
"""
import numba
import numpy as np

from .helper import (
    distance,
    scalar_product,
)
from .kernel_laplacian import KernelLaplacian


class DistanceKernel(KernelLaplacian):
    """
    Base class for dot-product kernels

    Specification to be implementation by children classes.
    """

    def __init__(self):
        super().__init__()
        self.kernel_type = "distance"

    def q_function(self, N, inplace: bool = True):
        r"""
        Function to compute dot-product kernel
        .. math::
            k(x, y) = q(\| x - y \|)

        To be implemented by children classes
        """
        raise NotImplementedError

    def q_function_derivative(self, N, inplace: bool = True):
        """
        Derivative of `q_function`.

        To be implemented by children classes
        """
        raise NotImplementedError

    def kernel(self, x1, x2):
        r"""
        Computation of the kernel

        Parameters
        ----------
        x1: ndarray of size (n, d)
        x2: ndarray of size (n, d)

        Returns
        -------
        K: ndarray of size (n, n)
        """
        N = distance(x1, x2)
        return self.q_function(N, inplace=True)

    def laplacian(self, x_repr, x, N=None, X=None, X_repr=None, D=None):
        r"""
        Computation of the discrete Laplacian operator
        .. math::
            L[i,j] = \sum_{k} \nabla_x k(x_k, y_i)^\top \nabla_x k(x_k, y_j)

        Parameters
        ----------
        x_repr: ndarray of size (p, d)
            ::math:`x_repr = y` representer to discretize `L^2`
        x: ndarray of size (n, d)
            Data to estimate the distribution on `L^2(X)`
        N: ndarray of size (p, n) (optional, default is None)
            Pre-computation of the distance matrix `x_repr.T @ x`
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
        if X is None:
            X = scalar_product(x_repr, x)
        if X_repr is None:
            X_repr = scalar_product(x_repr, x_repr)
        if D is None:
            D = np.sum(x**2, axis=1)
        inplace = False
        if N is None:
            N = distance(x_repr, x, scap=X)
            inplace = True
        Qprime = self.q_function_derivative(N, inplace=inplace)
        return self._laplacian(Qprime, X, X_repr, D)

    def nystrom(self, x_repr, x, N=None, K=None):
        r"""
        Computation of the discrete Nystrom operator
        .. math::
            R[i,j] = \sum_{k} k(x_k, y_i)^\top k(x_k, y_j)

        Parameters
        ----------
        x_repr: ndarray of size (p, d)
            ::math:`x_repr = y` representer to discretize `L^2`
        x: ndarray of size (n, d)
            Data to estimate the distribution on `L^2(X)`
        N: ndarray of size (p, n) (optional, default is None)
            Pre-computation of the dot-product matrix ::math:`\|x_r - x\|`
        K: ndarray of size (p, n) (optional, default is None)
            Pre-computation of the kernel `k(x_repr, x)`
        """
        if K is None:
            inplace = False
            if N is None:
                N = distance(x_repr, x)
                inplace = True
            K = self.q_function(N, inplace=inplace)
        R = K @ K.T
        return R

    @staticmethod
    @numba.jit("f8[:, :](f8[:, :], f8[:, :], f8[:, :], f8[:])", nopython=True)
    def _laplacian(Qprime, X, X_repr, D):
        p, n = Qprime.shape
        L = np.zeros((p, p), dtype=np.float64)
        for i in range(p):
            for j in range(p):
                for k in range(n):
                    norm = D[k] - 2 * X[i, k] + X_repr[i, i]
                    norm *= D[k] - 2 * X[j, k] + X_repr[j, j]
                    tmp = D[k] - X[i, k] - X[j, k] + X_repr[i, j]
                    if norm <= 0:
                        norm = 1
                        tmp = 0
                    L[i, j] += Qprime[i, k] * Qprime[j, k] * tmp / np.sqrt(norm)
        return L
