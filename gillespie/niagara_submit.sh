#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=24:00:00
#SBATCH --job-name gillespie-sir

#run this code using jbroths:~$ sbatch *script_name.sh*

# DIRECTORY TO RUN - $SLURM_SUBMIT_DIR is directory job was submitted from
cd $SLURM_SUBMIT_DIR

# load modules (must match modules used for compilation)
module load NiaEnv/2019b python/3.6.8
module load gnu-parallel

# Turn off implicit threading in Python, R
export OMP_NUM_THREADS=1

# EXECUTION COMMAND; ampersand off 40 jobs and wait
parallel --joblog slurm-$SLURM_JOBID.log -j $SLURM_TASKS_PER_NODE -N0 "python gillespie.py -m sir -t 100" ::: {1..100} --bar
