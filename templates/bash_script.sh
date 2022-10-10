#!/bin/bash

# Slurm bash script (sbatch).
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# @ 2022, Vivien Cabannes

# job configuration
#SBATCH --job-name=halfmoon
#SBATCH --time=00:05:00

# compute power
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G

# logging
#SBATCH --output=~/stdout-job-%j.out
#SBATCH --error=~/stderr-job-%j.err

# conda activate <env_name>
python ~/code/ssl/scripts/halfmoons.py -vvv -m Augmentation
python ~/code/ssl/scripts/halfmoons.py -vvv -m Laplacian
python ~/code/ssl/scripts/halfmoons.py -vvv -m Finite_diff
