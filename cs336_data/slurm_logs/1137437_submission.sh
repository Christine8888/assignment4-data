#!/bin/bash

# Parameters
#SBATCH --account=student
#SBATCH --cpus-per-task=1
#SBATCH --error=/home/c-cye/assignment4-data/cs336_data/slurm_logs/%j_0_log.err
#SBATCH --job-name=submitit
#SBATCH --mem=4GB
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/home/c-cye/assignment4-data/cs336_data/slurm_logs/%j_0_log.out
#SBATCH --partition=a4-cpu
#SBATCH --qos=a4-cpu-qos
#SBATCH --signal=USR2@90
#SBATCH --time=30
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home/c-cye/assignment4-data/cs336_data/slurm_logs/%j_%t_log.out --error /home/c-cye/assignment4-data/cs336_data/slurm_logs/%j_%t_log.err /home/c-cye/assignment4-data/.venv/bin/python3 -u -m submitit.core._submit /home/c-cye/assignment4-data/cs336_data/slurm_logs
