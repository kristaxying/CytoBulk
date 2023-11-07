#!/bin/bash
#SBATCH --job-name=run_sti
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=7-20:20:00
#SBATCH --output=TSA_%j.log
#SBATCH --partition work
source /home/wangxueying/miniconda3/bin/activate
conda activate cytobulk
python /data1/wangxueying/cytobulk/code/simulation_src.py
