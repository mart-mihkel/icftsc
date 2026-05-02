#!/usr/bin/env bash
#SBATCH --output=log/slurm/%j-%x.out
#SBATCH --gres=gpu:h200-141g:1
#SBATCH --cpus-per-task=16
#SBATCH --job-name="nvcc"
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --mem=128GB

MAX_JOBS=8 uv sync --compile-bytecode --extra cu128 --verbose
