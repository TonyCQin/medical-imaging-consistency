#!/bin/bash
#SBATCH -Jtest                    # Job name
#SBATCH --account=gts-smussmann3                 # charge account
#SBATCH -N1 --ntasks-per-node=1                 # Number of nodes and cores per node required
#SBATCH --gres=gpu:1 -C V100-32GB                 # gpu requests (v100, 1 gpu)
#SBATCH --mem-per-gpu=12G                        # Memory per gpu
#SBATCH -t00:30:00                             # Duration of the job 
#SBATCH -qembers                             # QOS Name
#SBATCH -oReport-test.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=tonyqin@gatech.edu        # E-mail address for notifications

module load anaconda3/2023.03                   # Load module dependencies
conda activate camera_traps
srun python test.py