#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16gb
#SBATCH --account=nexus
#SBATCH --partition=tron
#SBATCH --qos medium
#SBATCH -t 2-00:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH -o /fs/nexus-projects/PhysicsFall/smpl2motorica/slurm_output/%j.out

# activate conda environment
source /nfshomes/peng2000/.bashrc
conda activate python39
python train.py