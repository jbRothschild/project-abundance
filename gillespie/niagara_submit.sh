#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=12:00:00
#SBATCH --job-name gillespie-sir

#run this code using jbroths:~$ sbatch *script_name.sh*

# DIRECTORY TO RUN - $SLURM_SUBMIT_DIR is directory job was submitted from
cd $SLURM_SUBMIT_DIR

# load modules (must match modules used for compilation)
module load NiaEnv/2019b python/3.6.8
module load gnu-parallel

# Turn off implicit threading in Python, R
export OMP_NUM_THREADS=1

NUM_TASKS=40 # Generally 40, maybe more?
NUM_TASKS_ZERO=$((NUM_TASKS-1))

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

# multiLV model, varying parameters
VAR=($(logspace -2 0 ${NUM_TASKS} | tr -d '[],'))
VAR_NAME="comp_overlap"

parallel --joblog slurm-$SLURM_JOBID.log -j $SLURM_TASKS_PER_NODE "python gillespie.py -m multiLV -t 1 -n {#} -p ${VAR_NAME}={} max_gen_save=10000" ::: ${VAR[@]}
