#!/bin/bash
#SBATCH --partition=allgpu,maxgpu                    # for DESY maxwell users only
#SBATCH --constraint='A100'|'P100'|'V100'|
#SBATCH --time=72:00:00                           # Maximum time requested
#SBATCH --nodes=1                                 # Number of nodes
#SBATCH --chdir=<path-to-slurm-directory>         # directory must already exist!
#SBATCH --job-name=<job-name>
#SBATCH --output=%j.out                           # File to which STDOUT will be written
#SBATCH --error=%j.err                            # File to which STDERR will be written
#SBATCH --mail-type=NONE                          # Type of email notification- BEGIN,END,FAIL,ALL

# Set environment variables for wandb
export WANDB_PROJECT="<wandb-project-name>"
export WANDB_ENTITY="<wandb-entity>"
export WANDB__SERVICE_WAIT=300
unset LD_PRELOAD
source /etc/profile.d/modules.sh
module purge
module load maxwell gcc/9.3
module load anaconda3/5.2
. conda-init
conda activate <gen_model_playground environment>
cd <path-to-your-project>
python __main__.py --name gan #add args here