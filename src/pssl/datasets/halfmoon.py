"""
Half Moon dataset.

Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@ 2022, Vivien Cabannes
"""

from typing import Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class HalfMoonDataset(Dataset):
    """
    Half Moon dataset.

    Parameters
    ----------
    n: int, optional
        Number of data in the dataset. Default is 1000
    sigma: float, optional
        Noise level in the dataset. Default is zero.
    device: str, optional
        Computation device. Default is 'cpu'.
    dtype: type, optional
        Type of parameters. Default is torch.float (32 bits).
    complexe: bool, optional
        Either to create a dataset with four label corresponding to orthan.
        Default is False.
    """
    def __init__(self, n: int = 1000, sigma: float = 0,
                 device: str = 'cpu', dtype: type = torch.float, complex: bool = False):
        self.x, self.y = self.generate_half_moon(n, sigma, device, dtype, complex=complex)

    @staticmethod
    def generate_half_moon(n, sigma, device, dtype, complex=False):
        """
        Generate half moon dataset

        Returns
        -------
        x: ndarray
            Input data as design matrix of size `n` times 2.
        y: ndarray
            Output data in {0, 1} as a vector of size `n`.
        """
        theta = 2 * torch.pi * torch.randn(n, device=device, dtype=dtype)
        x = torch.empty((n, 2), device=device, dtype=dtype)
        x[:, 0] = torch.cos(theta)
        x[:, 1] = torch.sin(theta)
        y = (x[:, 0] > 0).type(dtype)
        if complex:
            z = (x[:, 1] > 0).type(dtype)
        x[y == 1, 1] += 1
        x += sigma * torch.randn(x.size(), device=device, dtype=dtype)
        if complex:
            y = F.one_hot((y + 2 * z).type(torch.LongTensor))
        return x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: Union[int, list[int]]):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x[idx], self.y[idx]
