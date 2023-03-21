import logging

import numpy as np
from scipy.linalg import eigh

from .helper import distance, scalar_product

logger = logging.getLogger("klap")
logger.setLevel(logging.INFO)


class KernelLaplacian:
    """
    Based class to estimate Laplacian through kernel methods

    Specification to be implementated by children classes.

    Parameters
    ----------
    p: int (optional, default=None)
        Number of representer points, if None, all points are used
    k: int (optional, default=16):
        Number of eigenvalues and eigenfunctions to estimate

    Attributes
    ----------
    eigenvalues: ndarray of size (k,)
        Eigenvalues of the Laplacian
    """

    def __init__(self, p=None, k=16):
        self.kernel_type = ""
        self.p = p
        self.k = k

    def kernel(self, x1, x2, **karwgs):
        r"""
        Computation of the kernel matrix
        .. math::
            K[i,j] = k(x1[i], x2[j])

        To be implemented by children classes

        Parameters
        ----------
        x1: ndarray of size (n, d)
        x2: ndarray of size (n, d)

        Returns
        -------
        K: ndarray of size (n, n)
        """
        raise NotImplementedError

    def laplacian(self, x_repr, x, **karwgs):
        r"""
        Computation of the discrete Laplacian operator
        .. math::
            L[i,j] = \sum_{k} \nabla_x k(x_k, y_i)^\top \nabla_x k(x_k, y_j)

        To be implemented by children classes

        Parameters
        ----------
        x_repr: ndarray of size (p, d)
            ::math:`x_repr = y` representer to discretize `L^2`
        x: ndarray of size (n, d)
            Data to estimate the distribution on `L^2(X)`

        Returns
        -------
        L: ndarray of size (p, p)
        """
        raise NotImplementedError

    def nystrom(self, x_repr, x, K=None):
        r"""
        Computation of the discrete Nystrom operator
        .. math::
            R[i,j] = \sum_{k} k(x_k, y_i)^\top k(x_k, y_j)

        To be implemented by children classes

        ----------
        x_repr: ndarray of size (p, d)
            ::math:`x_repr = y` representer to discretize `L^2`
        x: ndarray of size (n, d)
            Data to estimate the distribution on `L^2(X)`
        K: ndarray of size (p, n) (optional, default is None)
            Pre-computation of the kernel `k(x_repr, x)`
        """
        if K is None:
            K = self.kernel(x_repr, x)
        R = K @ K.T
        return R

    def fit(
        self,
        x,
        p=None,
        k=None,
        L_reg: float = 0,
        R_reg: float = 0,
        inverse_L: bool = False,
    ):
        """
        Estimate Laplacian operator based on data.

        Parameters
        ----------
        x: ndarray of size (n, d)
            Data matrix
        p: int (optional, default is None)
            Number of representer points to use
        k: int (optional, default is None)
            Number of eigenvalues to compute. If None, k will be taken as self.k (default is 16)
        L_reg: float (optional, default is 0)
            Regularization parameter for Laplacian matrix
        R_reg: float (optional, default is 0)
            Regularization parameter for Nystrom matrix
        inverse_L: bool (optional, default is False)
            Either to inverse L or R in the GEVD system.

        Returns
        -------
        lambdas: ndarray of size (k,)
            Eigenvalues of the Laplacian operator
        alphas: ndarray of size (p, k)
            Eigenvectors of the Laplacian operator
        """
        # Parsing arguemetns
        n = len(x)
        if p is None:
            if self.p is None:
                p = n
            else:
                p = self.p
        if k is None:
            k = self.k
        k = min(k, p - 1)

        # Selecting representer points
        self.x_repr = x[:p]

        # Building generalized eigenvalue problem
        if self.kernel_type == "dotproduct":
            logger.info("Dot product kernel computation")
            X = scalar_product(self.x_repr, x)
            X_repr = X[:, :p]
            L = self.laplacian(None, None, X=X, X_repr=X_repr)
            R = self.nystrom(None, None, X=X)
        elif self.kernel_type == "distance":
            logger.info("Distance kernel computation")
            X = scalar_product(self.x_repr, x)
            X_repr = X[:, :p]
            N = distance(self.x_repr, x, scap=X)
            D = np.sum(x**2, axis=1)
            L = self.laplacian(None, None, N=N, X=X, X_repr=X_repr, D=D)
            R = self.nystrom(None, None, N=N)
        elif self.kernel_type == "gaussian_fast":
            X = scalar_product(self.x_repr, x)
            K = self.kernel(self.x_repr, x, scap=X)
            X_repr = X[:, :p]
            D = np.sum(x**2, axis=1)
            L = self.laplacian(None, None, K=K, X=X, X_repr=X_repr, D=D)
            R = self.nystrom(None, None, K=K)
        else:
            logger.info("Non-spectific kernel computation")
            L = self.laplacian(self.x_repr, x)
            R = self.nystrom(self.x_repr, x)
        L /= n
        R /= n

        # Solving generalized eigenvalue problem
        if inverse_L:
            logging.info("Inversing L")
            error = eigh(L, eigvals_only=True, subset_by_index=[0, 0])[0]
            if error < 0:
                logger.info("Matrix is not sdp.")
                reg = -error * 10
                if L_reg < reg:
                    L_reg = reg
                logger.info(f"Setting regularizer to {L_reg:.3e}")
            L += L_reg * np.eye(p)
            R += R_reg * np.eye(p)
            lambdas, alphas = eigh(R, L, subset_by_index=[len(L) - k, len(L) - 1])
        else:
            logger.info("Inversing R")
            error = eigh(R, eigvals_only=True, subset_by_index=[0, 0])[0]
            if error < 0:
                logger.info("Matrix is not sdp.")
                reg = -error * 1.1
                if R_reg < reg:
                    R_reg = reg
                logger.info(f"Setting regularizer to {R_reg:.3e}")
            L += L_reg * np.eye(p)
            R += R_reg * np.eye(p)
            lambdas, alphas = eigh(L, R, subset_by_index=[0, k])

        # Storing results
        self.eigenvalues = lambdas
        self.alphas = alphas

    def features_map(self, x):
        """
        Compute the features map based on Laplacian eigenfunctions

        Parameters
        ----------
        x: ndarray of size (n, d)
            Data matrix

        Returns
        -------
        phi: ndarray of size (n, p)
            Features map
        """
        K = self.kernel(x, self.x_repr)
        return K @ self.alphas

    def __call__(self, *args, **kwargs):
        return self.features_map(*args, **kwargs)

    def diffusion_distance(self, x, y, t):
        """
        Compute the diffusion distance between two points
        ..math::
            d_t(x, y) = \sqrt{\sum_{i=1}^p \lambda_i^{2t} (\phi_i(x) - \phi_i(y))^2}

        Parameters
        ----------
        x: ndarray of size (n, d) or (d,)
            First point
        y: ndarray of size (n, d) or (d,)
            Second point
        t: float
            Time parameter

        Returns
        -------
        d: ndarray of size (n,) or float
            Diffusion distance
        """
        if len(x.shape) == 1:
            x = x[np.newaxis, :]
        if len(y.shape) == 1:
            y = y[np.newaxis, :]
        phi = self.features_map(x)
        phi -= self.features_map(y)
        phi *= self.lambdas[np.newaxis, :] ** t
        phi **= 2
        return np.sqrt(np.sum(phi, axis=1)).squeeze()
