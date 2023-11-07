#!/bin/bash
#SBATCH --job-name=run_methods
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=7-20:20:00
#SBATCH --output=TSA_%j.log
#SBATCH --partition work
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
Rscript /data1/wangxueying/cytobulk/code/st_other_deconv.R --rna_fn '/data1/wangxueying/cytobulk/eval_data/KIRC_GSE121636/KIRC_GSE121636_expression_test.csv' \
--meta_fn '/data1/wangxueying/cytobulk/eval_data/KIRC_GSE121636/meta_test.csv' \
--sc_fn '/data1/wangxueying/cytobulk/eval_data/KIRC_GSE121636/KIRC_GSE121636.h5Seurat' \
--out_dir '/data1/wangxueying/cytobulk/out/KIRC_GSE121636' \
--project 'KIRC_GSE121636' \
--method 'spatialdwls'

