#!/bin/bash
#SBATCH --job-name=run_methods
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=7-20:20:00
#SBATCH --output=TSA_%j.log
#SBATCH --partition work
source /home/wangxueying/miniconda3/bin/activate
conda activate cytobulk
Rscript /data1/wangxueying/cytobulk/code/st_other_deconv.R --rna_fn '/data1/wangxueying/cytobulk/eval_data/THCA_GSE148673_expression_test.csv' --sc_fn '/data1/wangxueying/cytobulk/eval_data/THCA_GSE148673.h5Seurat' --out_dir '/data1/wangxueying/cytobulk/out' --project 'THCA_GSE148673' --method 'card'
