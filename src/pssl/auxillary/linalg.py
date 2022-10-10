"""
Auxillary functions: linear algebra

Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@ 2022, Vivien Cabannes
"""

import torch


def diff_matrix(x: torch.Tensor, inplace: bool = False):
    """
    Compute matrix of squared distances

    .. math::
        y_{ij} = \| x_i - x_j \|^2

    Parameters
    ----------
    x: torch.Tensor
        Design matrix of size `number of samples * input dimension`.
    inplace: bool, optional
        Keyword to optimize some operations inplace. Default is False.

    Returns
    -------
    y: torch.Tensor
        Matrix of squared distances.
    """
    y = x @ x.transpose(1, 0)
    norm = torch.diag(y) / 2
    if inplace:
        # inplace operations, useful in no_grad mode
        y -= norm.unsqueeze(1)
        y -= norm
        y *= -1
    else:
        y = y - norm.unsqueeze(1)
        y = y - norm
        y = -y
    return y


def get_rbf_weights(x: torch.Tensor, sigma: float = 1):
    """
    Compute rbf weights matrix

    .. math::
        y_{ij} = \exp(-\| x_i - x_j \|^2 / \sigma^2)

    Parameters
    ----------
    x: torch.Tensor
        Design matrix of size `number of samples * input dimension`.
    sigma: float, optional
        Scale parameter. Default is 1.

    Returns
    -------
    y: torch.Tensor
        Gram matrix.
    """
    with torch.no_grad():
        y = diff_matrix(x, inplace=True)
        y /= - sigma ** 2
        y = torch.exp(y)
    return y
