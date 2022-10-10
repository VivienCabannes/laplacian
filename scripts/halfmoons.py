"""
Script to try data augmentation.

Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@ 2022, Vivien Cabannes
"""
# Import
import argparse
import datetime
import logging

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from pssl.architecture import MLP
from pssl.auxillary import (
    color_plot,
    training,
)
from pssl.config import (
    LOGS_PATH,
    SAVE_PATH,
    logging_config,
)
from pssl.datasets import HalfMoonDataset
from pssl.loss import (
    Dirichlet,
    augmentation_diff,
    graph_Laplacian,
    ortho_reg,
    ortho_reg_minibatch,
)


# Parsing arguments
# .. dataset parameters
parser = argparse.ArgumentParser(
    prog="HalfMoon",
    description="SSL Simulation with half-moon dataset",
    epilog="@ 2022, Vivien Cabannes",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-n",
    "--n_train",
    default=1000,
    nargs="?",
    help="Number of traning samples",
    type=int,
)
parser.add_argument(
    "-d",
    "--in_dim",
    default=2,
    nargs="?",
    help="Input dimension",
    type=int,
)
# .... scaling (dataset and methods)
parser.add_argument(
    "--sigma_noise",
    default=.1,
    nargs="?",
    help="Noise in the dataset",
    type=float,
)
parser.add_argument(
    "--sigma_augmentation",
    default=.1,
    nargs="?",
    help="Scale of augmentation.",
    type=float,
)
parser.add_argument(
    "--sigma_rbf",
    default=.1,
    nargs="?",
    help="Scale for rbf kernel in finite difference",
    type=float,
)
# .... internals
parser.add_argument(
    "--dtype",
    default=32,
    nargs="?",
    help="Floating point precision",
    type=int,
)
parser.add_argument(
    "--seed",
    default=0,
    nargs="?",
    help="Random seed",
    type=int,
)
# .. optimization parameter
parser.add_argument(
    "-b",
    "--batch_size",
    default=100,
    nargs="?",
    help="Batch size",
    type=int,
)
parser.add_argument(
    "-e",
    "--nb_epochs",
    default=int(5e2),
    nargs="?",
    help="Number of epochs",
    type=int,
)
parser.add_argument(
    "-l",
    "--lambda_reg",
    default=5e-0,
    nargs="?",
    help="Regularization parameter",
    type=float,
)
parser.add_argument(
    "-g",
    "--lr",
    default=5e-3,
    nargs="?",
    help="Learning rate",
    type=float,
)
# .... internal
parser.add_argument(
    "--momentum",
    default=0,
    nargs="?",
    help="Momemtum for SGD",
    type=float,
)
parser.add_argument(
    "--weigth_decay",
    default=0,
    nargs="?",
    help="Weigth decay in SGD",
    type=float,
)
parser.add_argument(
    "--gamma",
    default=0.3,
    nargs="?",
    help="Gamma parameter in learning rate scheduler",
    type=float,
)
# .. architecture
parser.add_argument(
    "-p",
    "--out_dim",
    default=5,
    nargs="?",
    help="Representation size (output dimension)",
    type=int,
)
parser.add_argument(
    "--mlp_units",
    default=[100, 100, 20],
    nargs="+",
    help="Hidden architecture",
    type=int,
)
# .. choice of method
parser.add_argument(
    "-m",
    "--method",
    default='Finite_diff',
    nargs="?",
    help="Method either 'Finite_diff', 'Laplacian', 'Augmentation' or 'Calibration'",
    type=str,
)
parser.add_argument(
    "-r",
    "--reg",
    default="Ortho",
    nargs="?",
    help="Regularization method, 'ortho' or 'minibatch'",
    type=str,
)
parser.add_argument(
    "--device",
    default={True: "cuda", False: "cpu"}[torch.cuda.is_available()],
    nargs="?",
    help="Device to run calculation, 'cuda' or 'cpu'",
    type=str,
)
# .. verbosity
parser.add_argument(
    "-v",
    "--verbosity",
    action="count",
    help="Increase in verbosity level",
    default=0,
)

# .. collecting arguments
config = parser.parse_args()


# Setting logging level
if config.verbosity == 0:
    logging_config['level'] = logging.CRITICAL
elif config.verbosity == 1:
    logging_config['level'] = logging.ERROR
elif config.verbosity == 2:
    logging_config['level'] = logging.WARNING
elif config.verbosity == 3:
    logging_config['level'] = logging.INFO
else:
    logging_config['level'] = logging.DEBUG

logging_path = LOGS_PATH / datetime.datetime.today().strftime("%Y-%m-%d.log")
logging_config["handlers"] = [
    logging.FileHandler(logging_path),
]

logging.basicConfig(**logging_config)
logger = logging.getLogger("pssl")
logger.critical("-" * 8 + "New script run" + "-" * 8)
logger.info(f"Parsing arguments: verbosity level = {config.verbosity}")
logger.debug(str(config))


# Global variables
DTYPE = {i: getattr(torch, "float" + str(i)) for i in [16, 32, 64]}[config.dtype]
DEVICE = config.device
NP_DTYPE = {i: getattr(np, "float" + str(i)) for i in [16, 32, 64]}[config.dtype]
logger.info(f"Runnning on {DEVICE} with precision {DTYPE}")


# Random seed
torch.random.manual_seed(config.seed)


# Regularizer
if config.reg.lower() == "ortho":
    reg = ortho_reg
elif config.reg.lower() == "minibatch":
    reg = ortho_reg_minibatch
else:
    raise NotImplementedError(f"Regularization {config.reg} not recognized")


# Objective
if config.method.lower() == "finite_diff":
    loss_function = graph_Laplacian
    requires_input_grad = False
elif config.method.lower() == "augmentation":
    loss_function = augmentation_diff
    requires_input_grad = False
elif config.method.lower() == "laplacian":
    loss_function = Dirichlet
    requires_input_grad = True
elif config.method.lower() == "calibration":
    def loss_function(net, inputs, outputs=None, **kwargs):
        return reg(outputs, centered=True)
else:
    raise NotImplementedError(f"Method {config.method} not recognized")


def main_training():
    """
    Main function
    """
    global dataset, net
    # Dataset loader
    nb_batch = config.n_train // config.batch_size
    dataset = HalfMoonDataset(
        n=config.n_train, sigma=config.sigma_noise, device=DEVICE, dtype=DTYPE
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Architecture and optimizer
    net = MLP(
        [config.in_dim, *config.mlp_units, config.out_dim], device=DEVICE, dtype=DTYPE
    )
    optimizer = optim.SGD(
        net.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weigth_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, config.nb_epochs * nb_batch / 3, gamma=config.gamma
    )

    # Training
    if config.method.lower() == "calibration":
        # To calibrate descent parameters: use the regularizer only.
        training(
            net,
            dataloader,
            loss_function,
            optimizer,
            loss_kwargs={
                "sigma_augmentation": config.sigma_augmentation,
                "sigma_rbf": config.sigma_rbf,
            },
            scheduler=scheduler,
            nb_epochs=config.nb_epochs,
        )
    else:
        training(
            net,
            dataloader,
            loss_function,
            optimizer,
            loss_kwargs={
                "sigma_augmentation": config.sigma_augmentation,
                "sigma_rbf": config.sigma_rbf,
            },
            regularizer=reg,
            scheduler=scheduler,
            requires_input_grad=requires_input_grad,
            lambda_reg=config.lambda_reg,
            nb_epochs=config.nb_epochs,
        )


main_training()


# Testing mode
net.eval()


def test_cov():
    """
    Checking orthogonality
    """
    with torch.no_grad():
        z = net(dataset.x)
        z = z - z.mean(dim=0)
        cov = z.transpose(0, 1) @ z / z.size(0)
        logger.info(f"covariance:\n{cov}")
        logger.info(ortho_reg(z, centered=True))


def test_representation():
    """
    Plotting learned functions
    """
    global X, Y, tox, x, y
    offset = 1.5
    X, Y = np.meshgrid(np.linspace(-offset, offset), np.linspace(-offset, offset + 1))
    x_test = np.vstack((X.flatten(), Y.flatten())).transpose().astype(NP_DTYPE)
    tox = torch.from_numpy(x_test).to(DEVICE)
    with torch.no_grad():
        z_test = net(tox).cpu().numpy()
        z_test -= np.mean(z_test, axis=0)

    x, y = dataset.x.cpu().numpy(), dataset.y.cpu().numpy()

    offset = 1.5
    for i in range(config.out_dim):
        savename = config.method.lower()[:3] + "_" + str(i)
        color_plot(
            X,
            Y,
            z_test[:, i],
            x,
            y,
            savename=savename,
            savepath=SAVE_PATH,
            figsize=(2, 1.5),
            alpha=0.1,
            s=1,
            vmin=-offset,
            vmax=offset,
            levels=20,
        )


def test_downstream():
    """
    Solving downstream task
    """
    n_test = 10000
    nb_exp = 100
    res = np.zeros(nb_exp)

    for i in range(nb_exp):
        with torch.no_grad():
            test_dataset = HalfMoonDataset(
                n=n_test,
                sigma=config.sigma_noise,
                device=DEVICE,
                dtype=DTYPE,
                complex=True,
            )
        phi, y_or = (
            net(test_dataset.x).detach().cpu().numpy(),
            test_dataset.y.cpu().numpy(),
        )
        w = np.linalg.solve(phi.T @ phi, phi.T @ y_or)
        res[i] = (np.argmax(phi @ w, axis=1) == np.argmax(y_or, axis=1)).mean()

    logger.info(f"{np.mean(res) * 100:.2f}, {np.std(res) * 100:.2f}")

    with torch.no_grad():
        phi_test = net(tox).cpu().numpy()
    z = np.argmax(phi_test @ w, axis=1)

    savename = config.method.lower()[:3] + "_test"
    color_plot(
        X,
        Y,
        z,
        x,
        y,
        savename=savename,
        savepath=SAVE_PATH,
        figsize=(2, 1.5),
        alpha=0.1,
        s=1,
        vmin=None,
        vmax=None,
        levels=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
    )


test_cov()
test_representation()
test_downstream()

logger.critical("Exiting script\n")
