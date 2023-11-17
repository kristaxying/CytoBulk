#!/bin/bash
#SBATCH --job-name=run_bulk
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=7-20:20:00
#SBATCH --output=TSA_%j.log
#SBATCH --partition work
source /home/wangxueying/miniconda3/bin/activate
conda activate cytobulk





Rscript /data1/wangxueying/cytobulk/code/bulk_other_deconv.R --bulk_rna '/data1/wangxueying/cytobulk/eval_data/human_sc/A35_sample_stimulated_bulk.txt' \
--sc_fn '/data1/wangxueying/cytobulk/eval_data/human_sc/filtered_A36_29_bulk.txt' \
--sc_meta '/data1/wangxueying/cytobulk/eval_data/human_sc/meta_A36_29_bulk.txt' \
--out_dir '/data1/wangxueying/cytobulk/out/human_sc' \
--project 'A36_sc_A35_bulk_1' \
--method 'SCDC'