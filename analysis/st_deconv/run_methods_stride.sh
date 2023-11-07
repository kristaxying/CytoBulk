#!/bin/bash
#SBATCH --job-name=run_stride
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=7-20:20:00
#SBATCH --output=TSA_%j.log
#SBATCH --partition work
source /home/wangxueying/miniconda3/bin/activate
conda activate spider
STRIDE deconvolve --sc-count /data1/wangxueying/cytobulk/eval_data/MM_GSE151310/MM_GSE151310_sc_data.txt \
--sc-celltype /data1/wangxueying/cytobulk/eval_data/MM_GSE151310/MM_GSE151310_sc_cell.txt \
--st-count /data1/wangxueying/cytobulk/eval_data/MM_GSE151310/MM_GSE151310_st_data.txt \
--outdir /data1/wangxueying/cytobulk/out/MM_GSE151310/STRIDE --outprefix MM_GSE151310



STRIDE deconvolve --sc-count /data1/wangxueying/cytobulk/eval_data/NSCLC_GSE179373/NSCLC_GSE179373_sc_data.txt \
--sc-celltype /data1/wangxueying/cytobulk/eval_data/NSCLC_GSE179373/NSCLC_GSE179373_sc_cell.txt \
--st-count /data1/wangxueying/cytobulk/eval_data/NSCLC_GSE179373/NSCLC_GSE179373_st_data.txt \
--outdir /data1/wangxueying/cytobulk/out/NSCLC_GSE179373/STRIDE --outprefix NSCLC_GSE179373



STRIDE deconvolve --sc-count /data1/wangxueying/cytobulk/eval_data/KIRC_GSE121636/KIRC_GSE121636_sc_data.txt \
--sc-celltype /data1/wangxueying/cytobulk/eval_data/KIRC_GSE121636/KIRC_GSE121636_sc_cell.txt \
--st-count /data1/wangxueying/cytobulk/eval_data/KIRC_GSE121636/KIRC_GSE121636_st_data.txt \
--outdir /data1/wangxueying/cytobulk/out/KIRC_GSE121636/STRIDE --outprefix KIRC_GSE121636



STRIDE deconvolve --sc-count /data1/wangxueying/cytobulk/eval_data/HNSC_GSE139324/HNSC_GSE139324_sc_data.txt \
--sc-celltype /data1/wangxueying/cytobulk/eval_data/HNSC_GSE139324/HNSC_GSE139324_sc_cell.txt \
--st-count /data1/wangxueying/cytobulk/eval_data/HNSC_GSE139324/HNSC_GSE139324_st_data.txt \
--outdir /data1/wangxueying/cytobulk/out/HNSC_GSE139324/STRIDE --outprefix HNSC_GSE139324

