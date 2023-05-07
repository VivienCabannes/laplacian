# Load libraries
import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np

from klap import (
    ExponentialKernel,
    GaussianKernel,
    PolynomialKernel,
)
from klap.utils import (
    SAVE_DIR,
    write_numpy_file,
)
from klap.datasets.helper import spherical_eigenvalues

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{mathtools}")
plt.rc("font", size=10, family="serif", serif="cm")
plt.rc("figure", figsize=(2, 1.5))

# Set logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "name",
    choices=[
        "exponential",
        "gaussian",
        "polynomial",
    ],
    help="Name of the experiment",
    type=str,
)
parser.add_argument(
    "--graph",
    action="store_true",
    help="Use graph Laplacian instead of fkernel Laplacian",
)
parser.add_argument(
    "-e",
    "--nb-exp",
    default=100,
    nargs="?",
    help="Number of experiments to estimate risk expectation",
    type=int,
)
parser.add_argument(
    "-k",
    "--num-eigenvalues",
    default=25,
    nargs="?",
    help="Number of eigenvalues to estimate",
    type=int,
)
config = parser.parse_args()


name = config.name
nb_trials = config.nb_exp
graph_laplacian = config.graph
K = config.num_eigenvalues

if graph_laplacian:
    save_file = SAVE_DIR / f"eigenvalues_{name}_graph.npy"
else:
    save_file = SAVE_DIR / f"eigenvalues_{name}.npy"

np.random.seed(100)
ns = np.logspace(1.5, 4, num=20).astype(int)
N = np.max(ns)
ps = np.logspace(1.5, 3, num=10).astype(int)

if name == "polynomial":

    def get_kernel(sigma):
        return PolynomialKernel(d=sigma)

    sigmas = [3, 4, 5, 6]

elif name == "exponential":

    def get_kernel(sigma):
        return ExponentialKernel(sigma=sigma)

    sigmas = [0.3, 1, 3, 10, 30, 100]

elif name == "gaussian":

    def get_kernel(sigma):
        return GaussianKernel(sigma=sigma)

    sigmas = [0.1, 0.3, 1, 3, 10, 30]

L_reg = 0
R_reg = 0
inverse_L = True
ground_truth = spherical_eigenvalues(K)
for num_trial in range(nb_trials):
    errors = np.zeros((len(ns), len(ps), len(sigmas)))
    X = np.random.randn(N, 3)
    X /= np.sqrt(np.sum(X**2, axis=1))[:, np.newaxis]
    for i, n in enumerate(ns):
        logging.info(f"Trial {num_trial + 1:3}, n={n:3}")
        x = X[:n]
        for j, p in enumerate(ps):
            if p > n:
                errors[i, j] = np.nan
                continue
            for k, sigma in enumerate(sigmas):
                kernel = get_kernel(sigma)
                if graph_laplacian:
                    weigth_kernel = GaussianKernel(sigma=1).kernel
                    kernel.fit_with_graph_laplacian(
                        weigth_kernel,
                        x,
                        p=p,
                        k=K,
                        L_reg=L_reg,
                        R_reg=R_reg,
                        inverse_L=inverse_L,
                    )
                else:
                    kernel.fit(
                        x, p=p, k=K, L_reg=L_reg, R_reg=R_reg, inverse_L=inverse_L
                    )
                kernel.eigenvalues -= ground_truth
                kernel.eigenvalues /= ground_truth + 1
                kernel.eigenvalues **= 2
                errors[i, j, k] = np.mean(kernel.eigenvalues)
                write_numpy_file(errors, save_file)
