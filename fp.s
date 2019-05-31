#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=JB_ML_JL_finalproject
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --output=slurm_%j.out

module purge
module load python3/intel/3.6.3

cd /scratch/jl860/fp
source venv/bin/activate
python -u main.py


