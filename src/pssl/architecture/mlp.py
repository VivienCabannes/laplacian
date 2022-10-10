"""
Multilayer perceptron.

Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@ 2022, Vivien Cabannes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Fully connected neural network (a.k.a. multilayer perceptron).

    With ReLU activation function (beside output layer).

    Parameters
    ----------
    sizes: list of int
        Number of neurons per layer.
        First element of the list corresponds to input size.
        Last element of the list corresponds to output size.
        Other elements specifies number of hidden neuron for each hidden layer.
    norm: str, optional
        Specify normalization layer. Either `batch` for batch norm, or `layer` for layer norm.
        Default is `'layer'`.
    device: str, optional
        Computation device. Default is `'cpu'`.
    dtype: type, optional
        Type of parameters. Default is None, falling back to torch.float (32 bits).
    """

    def __init__(
        self,
        sizes: list[int],
        layer_norm: str = "layer",
        device: str = "cpu",
        dtype: type = None,
    ):
        assert len(sizes) > 1, f"Sizes {sizes} should a list of more than two elements"
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        in_size = sizes[0]
        for i, out_size in enumerate(sizes[1:-1]):
            setattr(
                self, "fcs" + str(i), nn.Linear(in_size, out_size, **factory_kwargs)
            )
            if layer_norm.lower() == "layer":
                setattr(
                    self,
                    "nls" + str(i),
                    nn.LayerNorm(out_size, elementwise_affine=False, **factory_kwargs),
                )
            elif layer_norm.lower() == "batch":
                setattr(
                    self, "nls" + str(i), nn.BatchNorm1d(out_size, **factory_kwargs)
                )
            else:
                raise ValueError(f"Layer norm keyword is only supported for `batch` or `layer`, got {layer_norm}.")
            in_size = out_size
        self.hidden_layers = i + 1
        self.fco = nn.Linear(in_size, sizes[-1], **factory_kwargs)

    def forward(self, x: torch.Tensor):
        for i in range(self.hidden_layers):
            fc = getattr(self, "fcs" + str(i))
            nl = getattr(self, "nls" + str(i))
            x = fc(x)
            x = nl(x)
            x = F.leaky_relu(x)
        x = self.fco(x)
        return x

    def guillotine(self, x: torch.Tensor, n: int = 0):
        """
        Compute network pass until the last few layers.

        Parameters
        ----------
        n: int, optional
            Number of last layers to remove in forward pass.
            Default is 0, it remove the last fully connected layer.
        """
        n_layers = self.hidden_layers - n
        for i in range(n_layers):
            fc = getattr(self, "fcs" + str(i))
            x = fc(x)
            x = F.leaky_relu(x)
        return x
