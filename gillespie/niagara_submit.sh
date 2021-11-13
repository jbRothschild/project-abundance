#!/bin/bash
#SBATCH --nodes=43
#SBATCH --ntasks-per-node=40
#SBATCH --time=24:00:00
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

linspace () {
    start=$1
    end=$2
    if [ $# -ne 3 ]
    then
        count=11
    else
        count=$3
    fi

    python -c "import numpy as np; print(np.linspace($start, $end, $count).tolist())"
}

# EXECUTION COMMAND; ampersand off 40 jobs and wait

# SIR model
#parallel --joblog slurm-$SLURM_JOBID.log -j $SLURM_TASKS_PER_NODE "python gillespie.py -m sir -t 1000 -n {}" ::: `seq 0 ${NUM_TASKS_ZERO}`

NUM_TASKS_1=41 # Generally 40, maybe more?
NUM_TASKS_2=41
NUM_TASKS_ZERO=$((NUM_TASKS-1))

# multiLV model, varying parameters
VAR1=($(logspace -2 0 ${NUM_TASKS_1} | tr -d '[],'))
VAR1_NAME="comp_overlap"

VAR2=($(logspace -3 1 ${NUM_TASKS_2} | tr -d '[],'))
VAR2_NAME="immi_rate"

RESULTS_DIR='sim_results'
SIM_DIR='multiLV90'

mkdir -p ${RESULTS_DIR}/${SIM_DIR}

# 1 variable vary
#parallel --joblog slurm-$SLURM_JOBID.log -j $SLURM_TASKS_PER_NODE "python gillespie.py -m multiLV -t 1 -g 70000000 -n {#} -p ${VAR1_NAME}={} max_gen_save=10000 immi_rate=0.001 sim_dir=multiLV1" ::: ${VAR1[@]}

# 2 variable vary
#parallel --joblog slurm-$SLURM_JOBID.log --sshdelay 0.1 --wd $PWD "python gillespie.py -m multiLV -t 1 -g 1000000 -n {#} -p ${VAR1_NAME}={1} ${VAR2_NAME}={2} max_gen_save=10000 sim_dir=${SIM_DIR}" ::: ${VAR1[@]} ::: ${VAR2[@]}
parallel --joblog slurm-$SLURM_JOBID.log --sshdelay 0.1 --wd $PWD "python gillespie.py -m multiLV -t 1 -g 1000000000 -n {#} -p birth_rate=2.0 death_rate=1.0 carry_capacity=50 ${VAR1_NAME}={1} ${VAR2_NAME}={2} max_gen_save=10000000 sim_dir=${SIM_DIR}" ::: ${VAR1[@]} ::: ${VAR2[@]}


# 1 parameter choice only
#parallel --joblog slurm-$SLURM_JOBID.log --sshdelay 0.1 --wd $PWD "python gillespie.py -m multiLV -t 1 -g 30000000 -n {#} -p birth_rate=20.0 death_rate=1.0 immi_rate=0.001 comp_overlap=0.8689 carry_capacity=200 max_gen_save=10000 sim_dir=${SIM_DIR}" ::: {001..1600}
