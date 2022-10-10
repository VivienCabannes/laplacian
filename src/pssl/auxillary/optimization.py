"""
Auxillary functions: optimization

Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@ 2022, Vivien Cabannes
"""

import logging
from typing import Union

import torch

logger = logging.getLogger("aux-optim")


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    """
    Set learning rate online for optimizer

    Parameters
    ----------
    optimizer: torch.optim.Optimizer
        Optimizer object.
    lr: float
        Learning rates.
    """
    optimizer.param_groups[0]["lr"] = lr


def training(
    net: torch.nn.Module,
    dataloader: torch.utils.data.dataloader.DataLoader,
    loss_function,
    optimizer,
    loss_kwargs={},
    regularizer=None,
    lambda_reg: float = 1,
    nb_epochs: int = 1,
    scheduler=None,
    requires_input_grad: bool = False,
    verbose: bool = True,
    period_log: Union[None, int] = None,
    gain_log: int = 1
):
    """
    Wrapper to train a neural networks model.
    """
    if period_log is None:
        period_log = max(nb_epochs // 100, 1)
    use_reg = True
    if regularizer is None:
        use_reg = False

    for epoch in range(nb_epochs):
        local_verbose = False
        if verbose and epoch % period_log == period_log - 1:
            local_verbose = True
            run_loss, run_reg, run_obj, count = 0, 0, 0, 0

        for inputs, _ in dataloader:
            optimizer.zero_grad(set_to_none=True)
            if requires_input_grad:
                inputs.requires_grad = True

            # Objective computation
            outputs = net(inputs)
            loss = loss_function(net, inputs, outputs=outputs, **loss_kwargs)
            if use_reg:
                reg = regularizer(outputs, centered=True)
                obj = loss / lambda_reg + reg
            else:
                obj = loss
            obj.backward()

            # Gradient step
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if local_verbose:
                run_loss += loss.item()
                if use_reg:
                    run_reg += reg.item()
                    run_obj += obj.item()
                count += 1

        if local_verbose:
            run_obj *= gain_log
            if use_reg:
                logger.info(
                    f'[{epoch+1}] loss: {run_obj / count:.3f} = '
                    f'{run_loss / count:.3f} + {lambda_reg:.3f} * {run_reg / count:.3f}'
                )
            else:
                logger.info(f'[{epoch+1}] loss: {run_loss / count:.3f}')

    if verbose:
        logger.info('Finished Training')
