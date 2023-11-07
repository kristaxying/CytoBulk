#!/bin/bash
#SBATCH --job-name=run_methods
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=500G
#SBATCH --time=7-20:20:00
#SBATCH --output=TSA_%j.log
#SBATCH --partition highmem
source /home/wangxueying/miniconda3/bin/activate
conda activate cytobulk
Rscript /data1/wangxueying/cytobulk/code/st_other_deconv.R --rna_fn '/data1/wangxueying/cytobulk/eval_data/MM_GSE151310/MM_GSE151310_expression_test.csv' \
--meta_fn '/data1/wangxueying/cytobulk/eval_data/MM_GSE151310/meta_test.csv' \
--sc_fn '/data1/wangxueying/cytobulk/eval_data/MM_GSE151310/MM_GSE151310.h5Seurat' \
--out_dir '/data1/wangxueying/cytobulk/out/MM_GSE151310' \
--project 'MM_GSE151310' \
--method 'spotlight'

