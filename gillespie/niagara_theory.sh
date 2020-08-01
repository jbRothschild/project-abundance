#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=1:00:00
#SBATCH --job-name mlv-theory

#run this code using jbroths:~$ sbatch *script_name.sh*

# DIRECTORY TO RUN - $SLURM_SUBMIT_DIR is directory job was submitted from
cd $SLURM_SUBMIT_DIR

# load modules (must match modules used for compilation)
module load NiaEnv/2019b python/3.6.8
module load gnu-parallel

# Turn off implicit threading in Python, R
export OMP_NUM_THREADS=40

python analysis.py
