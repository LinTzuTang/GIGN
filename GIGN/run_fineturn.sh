#!/bin/bash
#SBATCH --job-name=gign_pdbbind_rna
#SBATCH --output=gign_pdbbind_rna_fine.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=72:00:00

module purge
module load conda cuda/12.4.1 
nvcc --version

conda activate gign

python finetuning_pdbbind.py




