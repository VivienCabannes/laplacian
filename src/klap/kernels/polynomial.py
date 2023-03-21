"""
Fast implementation of operator at play to estimate Laplacian eigendecomposition with polynomial kernel.

@ Vivien Cabannes, 2023
"""
from .dot_product_kernel import DotProductKernel


class PolynomialKernel(DotProductKernel):
    r"""
    Polynomial kernel.

    The polynomial kernel is defined as
    .. math::
        K[i,j] = (1 + x1[i, :]^\top x2[j,:])^d

    Parameters
    ----------
    d:
        Degree of the polynomial kernel
    """

    def __init__(self, d: int = 1):
        super().__init__()
        self.d = d

    def q_function(self, X, inplace: bool = True):
        r"""
        Function to compute the kernel from dot-product matrix.
        .. math::
            q(x) = (1 + x)^d

        Parameters
        ----------
        X: ndarray of size (n1, n2)
            Dot-product matrix
        inplace:
            If True, the computation is done inplace
        """
        if not inplace:
            X = X.copy()
        X += 1
        X **= self.d
        return X

    def q_function_derivative(self, X, inplace: bool = True):
        r"""
        Function to compute the derivative of the kernel from dot-product matrix.
        .. math::
            q'(x) = d (1 + x)^{d-1}

        Parameters
        ----------
        X: ndarray of size (n1, n2)
            Dot-product matrix
        inplace:
            If True, the computation is done inplace
        """
        if not inplace:
            X = X.copy()
        X += 1
        X **= self.d - 1
        X *= self.d
        return X
