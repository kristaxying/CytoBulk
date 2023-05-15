## install packages
library(Seurat)
library(ggplot2)
library(SCEVAN)
library(optparse)


if(F){
  opt <- list()
  opt$input <- '/data1/wangxueying/scRNA-seq/Seurat.rds'
}

run <- function(opt){
  RNA.res<-load(opt$input)
  results <- SCEVAN::pipelineCNA(RNA.res$RNA[["RNA"]]@counts,  par_cores = 6, SUBCLONES = TRUE, plotTree = TRUE)
  saveRDS(results,  '/data1/wangxueying/scRNA-seq/cnv.rds')
}

run(opt)