"""
Auxillary functions: plots

Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@ 2022, Vivien Cabannes
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def color_plot(
    X, Y, Z, x_train, y_train, savename=None, savepath=None,
    figsize=(2, 1.5), alpha=0.1, s=1, vmin=-1, vmax=1, levels=100,
):
    """
    Wrapper to generate color plot.
    """

    num = int(np.sqrt(Z.shape[0]))
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Represent contour of the learned function
    ax.contourf(
        X, Y, Z.reshape(num, num), cmap="RdBu_r", vmin=vmin, vmax=vmax, levels=levels
    )

    # Scatter datapoint
    for z in [0, 1]:
        ax.scatter(
            x_train[y_train == z, 0], x_train[y_train == z, 1], c="k", alpha=alpha, s=s
        )

    # Remove ticks parameters
    ax.tick_params(axis="both", which="major", labelleft=False, labelbottom=False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    # Save figure
    if savename is None:
        return
    if savepath is None:
        savepath = Path.cwd()
    if len(savename) < 4 or savename[-4] != ".":
        savename += ".pdf"
    fig.savefig(savepath / savename)
