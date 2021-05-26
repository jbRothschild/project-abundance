#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=1:00:00
#SBATCH --job-name =mlvi-2param-hours

#run this code using jbroths:~$ sbatch *script_name.sh*

# DIRECTORY TO RUN - $SLURM_SUBMIT_DIR is directory job was submitted from
cd $SLURM_SUBMIT_DIR

# load modules (must match modules used for compilation)
module load NiaEnv/2019b python/3.6.8 gnu-parallel

# Turn off implicit threading in Python, R
export OMP_NUM_THREADS=1

logspace () {
    start=$1
    end=$2
    if [ $# -ne 3 ]
    then
        count=10
    else
        count=$3
    fi

    python -c "import numpy as np; print(np.logspace($start, $end, $count).tolist())"
}

# EXECUTION COMMAND; ampersand off 40 jobs and wait

# SIR model
#parallel --joblog slurm-$SLURM_JOBID.log -j $SLURM_TASKS_PER_NODE "python gillespie.py -m sir -t 1000 -n {}" ::: `seq 0 ${NUM_TASKS_ZERO}`

NUM_TASKS_1=11 # Generally 40, maybe more?
NUM_TASKS_2=11
NUM_TASKS_ZERO=$((NUM_TASKS-1))

# multiLV model, varying parameters
VAR1=($(logspace -2 0 ${NUM_TASKS_1} | tr -d '[],'))
VAR1_NAME="comOverlap"

VAR2=($(logspace -3 1 ${NUM_TASKS_2} | tr -d '[],'))
VAR2_NAME="immiRate"

SIM_DIR='mehta'

# 2 variable vary
parallel --joblog slurm-$SLURM_JOBID.log --sshdelay 0.1 --wd $PWD "python run_langevin.py -s -id {#} -p ${VAR1_NAME}={1} ${VAR2_NAME}={2} -d ${SIM_DIR}" ::: ${VAR1[@]} ::: ${VAR2[@]}
