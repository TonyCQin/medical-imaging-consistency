#!/bin/bash
#SBATCH -JChexpert                    # Job name
#SBATCH --account=gts-smussmann3                 # charge account
#SBATCH -N1 --ntasks-per-node=1                 # Number of nodes and cores per node required
#SBATCH --gres=gpu:1 -C V100-32GB                 # gpu requests (v100, 1 gpu)
#SBATCH --mem-per-gpu=12G                        # Memory per gpu
#SBATCH -t8:00:00                             # Duration of the job 
#SBATCH -qembers                               # QOS Name
#SBATCH -oReport-baseline20%.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=tonyqin@gatech.edu        # E-mail address for notifications

module load anaconda3/2023.03                   # Load module dependencies
conda activate camera_traps
srun python chexpert.py