"""TODO import module that are needed:
Make sure not to run it as pytest suite"""

import numba
import numpy as np


@numba.jit("f8[:, :, :](f8[:, :], f8[:, :], f8[:, :], f8[:])")
def _rbf_laplacian_debug(K, S, S_in, S_out):
    p, n = K.shape
    out = np.zeros((p, p, n), dtype=np.float64)
    for i in range(p):
        for j in range(p):
            for k in range(n):
                tmp = S_out[k] - S[i, k] - S[j, k] + S_in[i, j]
                out[i, j, k] += K[i, k] * K[j, k] * tmp
    return out


@numba.jit("f8[:, :, :](f8[:, :], f8[:, :], f8[:, :], f8[:])")
def _exp_laplacian_debug(K, S, S_in, S_out):
    p, n = K.shape
    out = np.zeros((p, p, n), dtype=np.float64)
    for i in range(p):
        for j in range(p):
            for k in range(n):
                norm = (S_out[k] - 2*S[i, k] + S_in[i, i])
                norm *= (S_out[k] - 2*S[j, k] + S_in[j, j])
                tmp = S_out[k] - S[i, k] - S[j, k] + S_in[i, j]
                if norm < 0:
                    norm = 1
                    tmp = 0
                out[i, j, k] += K[i, k] * K[j, k] * tmp / np.sqrt(norm)
    return out

class myFunc:
    """Generate random function and check for derivatives calculation"""
    def __init__(self, d=1, sigma=1, n=10, kernel='gaussian'):
        self.x = np.random.randn(n, d)
        self.alpha = np.random.randn(n)
        self.sigma = sigma
        if kernel == 'gaussian':
            self.name = 'gaussian'
            self.kernel = rbf_kernel
            self.laplacian = _rbf_laplacian_debug
            self.norm = 2 / sigma ** 2
        else:
            self.name = 'exponential'
            self.kernel = exp_kernel
            self.laplacian = _exp_laplacian_debug
            self.norm = 1 / sigma


    @staticmethod
    def format(x):
        if type(x)!= np.ndarray:
            x = np.array(x)
        if len(x.shape) == 0:
            x = x[np.newaxis, np.newaxis]
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        return x

    def __call__(self, x):
        x = self.format(x) 
        k_x = self.kernel(x, self.x, sigma=self.sigma)
        out = k_x @ self.alpha
        return out.squeeze()

    def deriv(self, x):
        x = self.format(x)
        out = np.zeros((len(x), self.x.shape[1]))
        k = self.kernel(x, self.x, sigma=self.sigma)
        for i in range(self.x.shape[0]):
            tmp = self.x[i] - x
            if self.name == 'exponential':
                norm = np.sqrt(np.sum(tmp ** 2, axis=1)[:, np.newaxis])
                tmp /= norm
            out += tmp * k[:, i:i+1] * self.alpha[i]
        out *= self.norm
        return out.squeeze()
        
    def norm_gradient(self, x):
        out = self.deriv(x) 
        out **= 2
        if len(out.shape) > 1:
            out = np.sum(out, axis=1)
        return out

    def norm_gradient_laplacian(self, x):
        x = self.format(x)
        K = self.kernel(self.x, x, sigma=self.sigma)
        S_in = scalar_product(self.x, self.x)
        S = scalar_product(self.x, x)
        S_out = np.sum(x ** 2, axis=1)
        L = self.laplacian(K, S, (S_in + S_in.T) / 2, S_out)
        L *= self.norm ** 2
        L = (L + L.transpose(1, 0, 2)) / 2
        out = np.zeros(len(x))
        for i in range(len(self.alpha)):
            for j in range(len(self.alpha)):
                out += self.alpha[i] * self.alpha[j] * L[i, j]
        return out


def test_derivative():
    np.random.seed(0)
    d = 50
    f = myFunc(sigma=1, d=d, kernel='exponential')
    x = np.random.randn(1000, d)
    a = f.norm_gradient(x)
    b = f.norm_gradient_laplacian(x)
    assert (np.abs(a - b) / np.abs(a + b)).max() < 1e-7
