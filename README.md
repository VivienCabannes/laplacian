
# PSSL: A research library to explore principles behind self-supervised learning

## Examples
Examples are provided in the [scripts](scripts) folder.

## Requirements
PSSL is based on pytorch, GPU acceleration is leveraging CUDA.
You will therefore need to install pytorch and cuda drivers.
You may refer to pytorch website for installation directions.
Requirements are provided in the [`requirements.txt`](requirements.txt) files

## Building PSSL
To build PSSL, download the source code.
Using `git` and `ssh` this can be done by running the following command.
```shell
git clone --depth 1 git@github.com:fairinternal/PSSL.git
```
It can be built from the root folder with the python package installer
```shell
cd <Path to PSSL code>
pip install .
```
If you plan to use and develop it, install it in development mode
```shell
pip install -e .
```

## Installing PSSL
PSSL would soon be on the Python package index, which allows you to install it with the python installer.
```shell
pip install pssl
```

## How PSSL works
PSSL implements research ideas as in [Cabannes, Bietti and Balestriero 2022](LINK).
It is implemented with `pytorch`.

## Full documentation
Full documentation can be generated from docstrings with `sphinx`.


See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

License
-------
PSSL is MIT licensed, as found in the LICENSE file.
