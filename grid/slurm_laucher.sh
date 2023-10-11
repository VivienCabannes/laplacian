#!/usr/bin/bash

# Logging configuration
#SBATCH --job-name=eigenvals
#SBATCH --output=/checkpoint/vivc/eigenvals-poly-%a-%t.out
#SBATCH --error=/checkpoint/vivc/eigenvals-poly-%a-%t.err
#SBATCH --mail-type=END
#SBATCH --mail-user=vivc@meta.com

# Job specification
#SBATCH --partition=devlab
#SBATCH --time=72:00:00
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=0
#SBATCH --array=1-500


python /private/home/vivc/code/laplacian/grid/run.py --num-tasks $SLURM_ARRAY_TASK_COUNT --task-id $SLURM_ARRAY_TASK_ID --name eigenvals --kernel polynomial
