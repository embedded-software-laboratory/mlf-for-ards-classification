#!/bin/bash

# Request tasks per node. In this case we request 1 node and 10 cpu cores on this node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

# Request memory per thread. Here it's 10 GB per requested core
#SBATCH --mem-per-cpu=10G

# Set a job name
#SBATCH --job-name=framework

# Specify time before the job is killed by scheduler (in case it hangs). In this case - 5.5 hour. NOTE - jobs with lower time are prefered by the scheduler and will wait less time to get picked up in case of overloading of the system
#SBATCH --time=5:30:00

# Declare the merged STDOUT/STDERR file. NOTE - make sure the directory specified (if you have specified one) actually exists
#SBATCH --output=./Runscripts/output_framework

# Send emails
#SBATCH --mail-user=example@example.com
#SBATCH --mail-type=END,FAIL

# Run on a specified project to not use your own computation time
#SBATCH --account=<account_name_here>
module load GCCcore/.12.2.0
export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate ai
python main.py
