
KLAP: Continuous (kernel) Laplacian spectral embedding
======================================================

:Topic: Fast Implementation of Laplacian eigenvectors and eigenvalues estimation with kernel methods
   developped in [PIL20]_, [CAB21]_, [PIL23]_.
:Author: Vivien Cabannes
:Version: 0.0.1 of 2023/02/21

Installation
------------
From wheel
~~~~~~~~~~
You can download our package from its pypi repository.

.. code:: shell

   $ pip install klap

From source
~~~~~~~~~~~
You can download source code at https://github.com/VivienCabannes/laplacian/archive/master.zip.
Once download, our packages can be install through the following command.

.. code:: shell

   $ cd <path to code folder>
   $ pip install .

Usage
-----
See `notebooks` folder.

Package Requirements
--------------------
Most of the code is based on the following python libraries:
 - numpy
 - numba
 
Testing done with notebook are based on:
 - jupyter-notebook
 - matplotlib
 - scipy

The code could easily be rewritten for pytorch (with jit support).
For generalized eigenvalues decomposition, see `torch.lobpcg`.

References
----------
.. [PIL20] Statistical estimation of the poincaré constant and application to sampling multimodal distributions, 
   Loucas Pillaud-Vivien et al., *AISTATS*, 2020

.. [CAB21] Overcoming the curse of dimensionality with Laplacian regularization
   in semi-supervised learning, Cabannes et al., *NeurIPS*, 2021

.. [PIL23] Kernelized Diffusion maps, 
   Loucas Pillaud-Vivien et al., *ArXiv*, 2023