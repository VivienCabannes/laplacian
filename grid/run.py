"""
Script to launch scaling laws experiments
"""
import logging
import time
from pathlib import Path

import numpy as np

from klap import (
    ExponentialKernel,
    GaussianKernel,
    PolynomialKernel,
)
from klap.utils import SAVE_DIR
from klap.utils.io import NpEncoder


# Logging
logger = logging.getLogger("script")
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="{asctime} {levelname} [{filename}:{lineno}] {message}",
    style="{",
    datefmt="%H:%M:%S",
    level="INFO",
    handlers=[
        logging.StreamHandler(),
    ],
)


def run_exp(config):
    """
    Run an experiment with the given configuration

    Parameters
    ----------
    config: argparse.Namespace
        configuration

    Returns
    -------
    res: dict
        results
    """
    res = {
        "n": config.n,
        "d": config.d,
        "p": config.p,
        "kernel": config.kernel,
        "kernel_param": config.kernel_param,
        "graph_laplacian": config.graph_laplacian,
        "seed": config.seed,
    }
    logging.info(repr(res))

    # Catching bad configurations
    if config.p > config.n:
        logging.warning("p > n, skipping configuration")
        res["eigenvalues"] = np.full(config.num_eigenvals, np.nan)
        res["time"] = 0
        return res

    np.random.seed(config.seed)

    # Data generation
    x = np.random.randn(config.n, config.d)
    x /= np.linalg.norm(x, axis=-1, keepdims=True)

    # Function for Galerkin methods
    if config.kernel == "exponential":
        kernel = ExponentialKernel(sigma=config.kernel_param)
    elif config.kernel == "gaussian":
        kernel = GaussianKernel(sigma=config.kernel_param)
    elif config.kernel == "polynomial":
        kernel = PolynomialKernel(d=config.kernel_param)
    else:
        raise NotImplementedError(f"No implementation for kernel: {config.kernel}")

    # Graph Laplacian
    if config.graph_laplacian:
        if config.graph_laplacian == -1:
            lap_ker = kernel.kernel
        else:
            lap_ker = GaussianKernel(sigma=config.graph_laplacian).kernel

    # Eigenfunctions estimation
    t = time.time()
    if config.graph_laplacian:
        kernel.fit_with_graph_laplacian(
            lap_ker,
            x,
            p=config.p,
            k=config.num_eigenvals,
            L_reg=0,
            R_reg=0,
            inverse_L=True,
        )
    else:
        kernel.fit(
            x, p=config.p, k=config.num_eigenvals, L_reg=0, R_reg=0, inverse_L=True
        )
    t = time.time() - t

    res["eigenvalues"] = kernel.eigenvalues
    res["time"] = t
    return res


if __name__ == "__main__":

    import argparse
    from itertools import product
    import json
    import sys

    # Configuration

    parser = argparse.ArgumentParser(
        description="Laplacian eigenvalues estimation",
    )
    sset = parser.add_argument_group("Settings")
    sset.add_argument(
        "-n",
        "--n",
        default=1000,
        type=int,
        help="number of training data",
    )
    sset.add_argument(
        "-d",
        "--d",
        default=3,
        type=int,
        help="dimension of input data",
    )
    sset.add_argument(
        "-k",
        "--num-eigenvals",
        default=25,
        type=int,
        help="number of eigenvalues to estimate",
    )
    mset = parser.add_argument_group("Methods")
    mset.add_argument(
        "-m",
        "--kernel",
        type=str,
        choices=["exponential", "gaussian", "polynomial"],
        default="polynomial",
        help="kernel for eigenfunctions recontsruction",
    )
    mset.add_argument(
        "-s",
        "--kernel-param",
        type=float,
        default=10,
        help="kernel parameters",
    )
    mset.add_argument(
        "-p",
        "--p",
        default=100,
        type=int,
        help="number of Galerkin repreenters",
    )
    mset.add_argument(
        "-g",
        "--graph-laplacian",
        default=0,
        type=float,
        help="scale for graph Laplacian - 0 = no graph-Laplacian, -1 = 'kernel-param'",
    )
    rset = parser.add_argument_group("Reproducibility")
    rset.add_argument(
        "--seed",
        default=42,
        type=int,
        help="random seed for reproducibility and diversity",
    )
    rset.add_argument(
        "--interactive",
        action="store_true",
        help="interactive mode (in constrast with grid run)",
    )
    rset.add_argument(
        "--save-dir",
        default=SAVE_DIR,
        help="saving directory",
    )
    rset.add_argument(
        "--save-name",
        default="test",
        help="folder name for saving",
    )
    rset.add_argument(
        "--num-tasks",
        default=100,
        type=int,
        help="number of tasks to split the grid runs into",
    )
    rset.add_argument(
        "--task-id",
        default=1,
        type=int,
        help="task id, from 1 to num_tasks",
    )
    config = parser.parse_args()

    # Interactive mode
    if config.interactive:
        res = run_exp(config)
        print("Results:")
        for k, v in res.items():
            print("\t", k, v)
        sys.exit(0)

    # Grid runs
    grid = {
        "n": np.unique(np.logspace(2, 5, num=10).astype(int)),
        "d": np.arange(3, 20),
        "p": np.unique(np.logspace(1, 3, num=5).astype(int)),
        "seed": list(range(42, 52)),
    }

    if config.kernel == "polynomial":
        grid.update({
            "graph_laplacian": [0, 0.01, 0.1, 1, 10, 100],
            "kernel": ["polynomial"],
            "kernel_param": [2, 3, 4, 5, 6],
        })
    elif config.kernel == "exponential":
        grid.update({
            "graph_laplacian": [-1, 0, 0.01, 0.1, 1, 10, 100],
            "kernel": ["exponential"],
            "kernel_param": [0.01, 0.1, 1, 10, 100],
        })
    elif config.kernel == "gaussian":
        grid.update({
            "graph_laplacian": [-1, 0, 0.01, 0.1, 1, 10, 100],
            "kernel": ["gaussian"],
            "kernel_param": [0.01, 0.1, 1, 10, 100],
        })

    # Output file
    outdir = Path(config.save_dir) / config.save_name / config.kernel
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"task_{config.task_id}.jsonl"

    # Clean file
    with open(outfile, "w") as f:
        pass

    # Run grids
    for i, vals in enumerate(product(*grid.values())):
        # Splitting the grid into tasks
        if i % config.num_tasks != (config.task_id - 1):
            continue
        # Setting configuration
        for k, v in zip(grid.keys(), vals):
            setattr(config, k, v)
        # Running experiment
        res = run_exp(config)
        # Saving results
        with open(outdir / f"task_{config.task_id}.jsonl", "a") as f:
            print(json.dumps(res, cls=NpEncoder), file=f)
