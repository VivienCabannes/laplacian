"""
Loss functions.

Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@ 2022, Vivien Cabannes
"""

from typing import Union

import torch
import torch.nn as nn
import torch.autograd as autograd
from .auxillary.linalg import (
    diff_matrix,
    get_rbf_weights,
)


def ortho_reg(x: torch.Tensor, centered=False):
    """
    Orthogonal features penalization.

    .. math::
        y = \| E[x x^\top] - I \|^2

    Parameters
    ----------
    x: torch.Tensor
        Design matrix of size `number of samples * number of dimension`.
    centered: bool, optional
        Either to center `x` before computing covariance. Default is False.

    Returns
    -------
    loss: torch.Tensor
        Empirical orthogonal regularization loss.
    """
    if centered:
        x = x - x.mean(dim=0)
    x = x.transpose(1, 0) @ x / x.size(0)
    x = x - torch.eye(x.size(0), device=x.device)
    loss = torch.sum(x ** 2)
    return loss


def ortho_reg_minibatch(x: torch.Tensor, centered=False):
    """
    Orthogonal features penalization implemented to provide unbiased stochastic gradient.

    .. math::
        y = E[(x^\top x')^2 - 2 E[(x^\top x)] + p

    Parameters
    ----------
    x: torch.Tensor
        Design matrix of size `number of samples * number of dimension` (`2n * p`).
    centered: bool, optional
        Either to center `x` before computing covariance. Default is False.

    Returns
    -------
    loss: torch.Tensor
        Empirical orthogonal regularization loss.
    """
    if centered:
        x = x - x.mean(dim=0)
    n = x.size(0) // 2
    if n < 1:
        raise ValueError(f'Batch size should be at least two, got {x.size(0)}.')
    # Quadratic part
    y = x[:n] @ x[n:].transpose(1, 0)
    loss = torch.mean(y ** 2)
    # Linear part
    loss = loss - torch.mean(x ** 2) * 2 * x.size(1)
    loss = loss + x.size(1)
    return loss


def Dirichlet(module: nn.Module, inputs: torch.Tensor, **kwargs):
    """
    Compute Dirichlet energy of module on inputs.

    .. math::
        l = \frac{1}{npd}\sum_{i < n, j < p} \| \nabla f_j(x_i) \|^2

    Parameters
    ----------
    net: torch.nn.Module
        Pytorch neural network module, producing an output of size `p`.
    inputs: torch.Tensor
        Tensor of inputs to feed the networks of size `n * d`.

    Returns
    -------
    loss: torch.Tensor
        Empirical Dirichlet energy up to dimensional constants.
    """
    def to_diff(x):
        return torch.sum(module(x), dim=0)
    jac = autograd.functional.jacobian(to_diff, inputs, create_graph=True)
    loss = torch.mean(jac ** 2)
    return loss


def augmentation_diff(net: nn.Module, inputs: torch.Tensor, outputs: Union[None, torch.Tensor] = None,
                      sigma_augmentation: float = 1, **kwargs):
    """
    Compute average difference between inputs and augmented inputs.

    Parameters
    ----------
    net: torch.nn.Module
        Pytorch neural network module.
    inputs: torch.Tensor
        Tensor of inputs to feed the networks.
    outputs: torch.Tensor, optional
        To avoid recomputing outputs inside function scope. Default is None.
    sigma_augmentation: float, optional
        Noise level in augmentation. Standard deviation of the Gaussian.

    Returns
    -------
    loss: torch.Tensor
        Empirical average of augmentation differences.
    """
    factory_kwargs = {'device': inputs.device, 'dtype': inputs.dtype}

    if outputs is None:
        outputs = net(inputs)

    aug = torch.randn(inputs.size(), **factory_kwargs)
    aug = sigma_augmentation * aug
    aug = inputs + aug
    aug_out = net(aug)

    diff = outputs - aug_out
    diff = diff ** 2
    loss = torch.mean(diff)
    return loss


def graph_Laplacian(net: nn.Module, inputs: torch.Tensor, outputs: Union[None, torch.Tensor] = None,
                    sigma_rbf: float = 1, **kwargs):
    """
    Compute graph Laplacian criterion on data.

    Parameters
    ----------
    net: torch.nn.Module
        Pytorch neural network module.
    inputs: torch.Tensor
        Tensor of inputs to feed the networks.
    outputs: torch.Tensor, optional
        To avoid recomputing outputs inside function scope. Default is None.
    sigma_rbf: float, optional
        Scale parameter in the graph Laplacian with rbf kernel.

    Returns
    -------
    loss: torch.Tensor
        Loss function based on the graph Laplacian.
    """
    if outputs is None:
        outputs = net(inputs)

    diff = diff_matrix(outputs)
    sim_weights = get_rbf_weights(inputs, sigma=sigma_rbf)
    diff = sim_weights * diff
    loss = torch.mean(diff)
    return loss
