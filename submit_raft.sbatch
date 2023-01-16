#!/bin/bash

#SBATCH -n 16
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --job-name="raftba_1_1"
#SBATCH --mem-per-cpu=1024

python raftba.py