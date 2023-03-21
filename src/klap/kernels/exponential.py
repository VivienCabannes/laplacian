"""
Fast implementation of operator at play to estimate Laplacian eigendecomposition with exponential kernel.

@ Vivien Cabannes, 2023
"""
import numpy as np

from .distance_kernel import DistanceKernel


class ExponentialKernel(DistanceKernel):
    r"""
    Exponential kernel.

    The exponential kernel is defined as
    .. math::
        K[i,j] = \exp(-\| (x1[i, :] - x2[j,:]) / \sigma \|)

    Parameters
    ----------
    sigma:
        Bandwidth parameter for the kernel
    """

    def __init__(self, sigma: float = 1):
        super().__init__()
        self.sigma = sigma

    def q_function(self, N, inplace: bool = True):
        r"""
        Function to compute the kernel from distance matrix.
        .. math::
            q(x) = exp(-x / \sigma)

        Parameters
        ----------
        N: ndarray of size (n1, n2)
            Distance matrix
        inplace:
            If True, the computation is done inplace
        """
        if not inplace:
            N = N.copy()
        N *= -1
        N /= self.sigma
        np.exp(N, out=N)
        return N

    def q_function_derivative(self, N, inplace: bool = True):
        r"""
        Function to compute the derivative of the kernel from distance matrix.
        .. math::
            q'(x) = -exp(-x / \sigma) / \sigma

        Parameters
        ----------
        X: ndarray of size (n1, n2)
            Distance matrix
        inplace:
            If True, the computation is done inplace
        """
        out = self.q_function(N, inplace=inplace)
        out *= -1
        out /= self.sigma
        return out
