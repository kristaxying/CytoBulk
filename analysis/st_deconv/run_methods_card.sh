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



Rscript /data1/wangxueying/cytobulk/code/st_other_deconv.R --rna_fn '/data1/wangxueying/cytobulk/sti/LSCC_GSE150321_spot/training_data/LSCC_GSE150321_spot_expression_test.csv' \
--meta_fn '/data1/wangxueying/cytobulk/eval_data/MM_GSE151310/meta_test.csv' \
--sc_fn '/data1/wangxueying/cytobulk/eval_data/LSCC_GSE150321_spot/LSCC_GSE150321_spot.h5Seurat' \
--out_dir '/data1/wangxueying/cytobulk/out/LSCC_GSE150321_spot' \
--project 'LSCC_GSE150321_spot' \
--method 'card'