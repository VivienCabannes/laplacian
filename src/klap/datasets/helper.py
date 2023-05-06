"""
Helpers functions

@ Vivien Cabannes, 2023
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sph_harm


# Spherical helper functions
def spherical_harmonic(x, num, return_eigenvalues=False, dtype=float):
    out = np.zeros((len(x), num), dtype=dtype)
    phi = np.arccos(x[:, 2])
    theta = np.arctan2(x[:, 1], x[:, 0])
    i = 0
    eigenvalues = np.zeros(num)
    for i in range(num):
        deg = int(np.sqrt(i))
        freq = i - deg**2 - deg
        if dtype == float:
            out[:, i] = np.real(sph_harm(freq, deg, theta, phi))
        else:
            out[:, i] = sph_harm(freq, deg, theta, phi)
        eigenvalues[i] = deg * (deg + 1)
    if return_eigenvalues:
        return out, eigenvalues
    return out


def spherical_eigenvalues(k):
    L = np.ceil(np.sqrt(k)).astype(int)
    eigenvalues = np.zeros(L**2)
    for degree in range(L):
        eigenvalues[degree ** 2:(degree + 1) ** 2] = degree * (degree + 1)
    eigenvalues = eigenvalues[:k]
    return eigenvalues


def meshgrid_3d(n):
    theta = np.linspace(0, 2 * np.pi, n)
    phi = np.linspace(0, np.pi, n)
    X = np.outer(np.cos(theta), np.sin(phi))
    Y = np.outer(np.sin(theta), np.sin(phi))
    Z = np.outer(np.ones(n), np.cos(phi))
    return X, Y, Z


def plot_sphere_surface(ax, X, Y, Z, f, cmap="RdBu", **kwargs):
    cm = plt.get_cmap(cmap)(f.reshape(*X.shape))
    ax.plot_surface(X, Y, Z, facecolors=cm, **kwargs)
    ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([-0.8, 0.8])
    ax.set_zlim([-0.8, 0.8])
    ax.set_axis_off()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])
