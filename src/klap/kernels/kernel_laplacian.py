import numpy as np
from scipy.linalg import eigh


class KernelLaplacian:
    def __init__(self, p=None, k=16):
        self.p = p
        self.k = k

    def fit(self, x, p=None, k=None, verbose=False):
        """
        Estimate Laplacian operator based on data.

        Parameters
        ----------
        x: ndarray of size (n, d)
            Data matrix
        p: int (optional, default is None)
            Number of representer points to use
        k: int (optional, default is None)
            Number of eigenvalues to compute
        verbose: bool (optional, default is False)
            If True, the eigenvalues are returned

        Returns
        -------
        lambdas: ndarray of size (k,)
            Eigenvalues of the Laplacian operator
        alphas: ndarray of size (p, k)
            Eigenvectors of the Laplacian operator
        """
        n = len(x)
        if p is None:
            if self.p is None:
                p = n
            else:
                p = self.p
        self.x_repr = x[:p]
        L = self.laplacian(self.x_repr, x)
        R = self.nystrom(self.x_repr, x)
        L /= n
        R /= n

        # Regularization
        error = eigh(L, eigvals_only=True, subset_by_index=[0, 0])[0]
        if error < 0:
            # numerical leads to negative eigenvalues
            reg = -error * 1.1
        if error > 0:
            reg = 1e-7
        L += reg * np.eye(len(L))
        lambdas, alphas = eigh(R, L, subset_by_index=[len(L) - k, len(L) - 1])
        self.lambdas = lambdas
        self.alphas = alphas
        if verbose:
            return lambdas, alphas

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
