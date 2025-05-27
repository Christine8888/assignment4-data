#!/bin/bash
#SBATCH --job-name=training
#SBATCH --output=training_%j.log
#SBATCH --error=training_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=a4-batch-qos
#SBATCH --partition=a4-batch

uv run torchrun --standalone --nproc_per_node=2 scripts/train.py --config-name=experiment/your_data