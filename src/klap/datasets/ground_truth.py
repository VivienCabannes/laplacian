import numpy as np
from scipy.special import binom


def spherical_eigenvals(d, k):
    eigenvals = np.empty(k, dtype=float)
    eigenvals[0] = 1
    i, s = 1, 1
    while True:
        N = binom(d + s - 3, s - 1)
        N *= (2 * s + d - 2) / s
        N = int(N)
        lambd = s * (s + d - 2)
        if N + i > k:
            eigenvals[i:] = lambd
            break
        eigenvals[i:i + N] = lambd
        i += N
        s += 1
    eigenvals **= -1
    eigenvals[0] = 0
    return eigenvals
