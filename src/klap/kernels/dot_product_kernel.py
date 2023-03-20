"""
Fast implementation of operator at play to estimate Laplacian eigendecomposition with dot-product kernel.

@ Vivien Cabannes, 2023
"""
from .helper import scalar_product
from .kernel_laplacian import KernelLaplacian


class DotProductKernel(KernelLaplacian):
    def __init__(self):
        pass

    def q_function(self, *args, **kwargs):
        raise NotImplementedError

    def q_function_derivative(self, *args, **kwargs):
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
        X = scalar_product(x1, x2)
        return self.q_function(X)

    def laplacian(self, x_repr, x, X=None, X_repr=None):
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
        X: ndarray of size (p, n) (optional, default is None)
            Pre-computation of the dot-product matrix `x_repr.T @ x`
        X_repr: ndarray of size (p, p) (optional, default is None)
            Pre-computation of the dot-product matrix `x_repr.T @ x_repr`

        Returns
        -------
        L: ndarray of size (p, p)
        """
        if X is None:
            X = scalar_product(x_repr, x)
        Qprime = self.q_function_derivative(X)
        L = Qprime @ Qprime.T
        if X_repr is None:
            X_repr = scalar_product(x_repr, x_repr)
        L *= X_repr
        return L

    def nystrom(self, x_repr, x, X=None, K=None):
        r"""
        Computation of the discrete Nystrom operator
        .. math::
            R[i,j] = \sum_{k} k(x_k, y_i)^\top k(x_k, y_j)

        ----------
        x_repr: ndarray of size (p, d)
            ::math:`x_repr = y` representer to discretize `L^2`
        x: ndarray of size (n, d)
            Data to estimate the distribution on `L^2(X)`
        X: ndarray of size (p, n) (optional, default is None)
            Pre-computation of the dot-product matrix `x_repr.T @ x`
        K: ndarray of size (p, n) (optional, default is None)
            Pre-computation of the gaussian kernel `k(x_repr, x)`
        """
        if K is None:
            if X is None:
                X = scalar_product(x_repr, x)
            K = self.q_function(X)
        R = K @ K.T
        return R
