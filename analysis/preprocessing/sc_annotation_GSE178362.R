## install packages
library(MAESTRO)
library(Seurat)
library(ggplot2)
library(presto)
library(Gmisc)
library(SCEVAN)


source("D:/project/deconvolution_algorithm/coding/Spatial_mapping/CytoBulk/analysis/preprocessing/sc_utils.R")

expr = Read10X_h5("D:/project/deconvolution_algorithm/coding/Spatial_mapping/data/GSE178362/sc/GSM5388413_DD073R_filtered_feature_bc_matrix.h5")
# for multiple samples
tes_endt<-readRDS("C:/Users/xywang85/Downloads/GSE178360_endo.sub_SAE10x3.integrated_temp_v2 (1).RDS/GSE178360_endo.sub_SAE10x3.integrated_temp_v2.RDS")
tes_me<-readRDS("C:/Users/xywang85/Downloads/GSE178360_mes.sub_SAE10x3.integrated_v2.RDS/GSE178360_mes.sub_SAE10x3.integrated_v2.RDS")
tes_immu<-readRDS("C:/Users/xywang85/Downloads/GSE178360_immune_platelet.sub_SAE10x3.integrated_temp.RDS/GSE178360_immune_platelet.sub_SAE10x3.integrated_temp.RDS")
tes_epi<-readRDS("C:/Users/xywang85/Downloads/GSE178360_epi.sub_SAE10x3.integrated_v2.RDS/GSE178360_epi.sub_SAE10x3.integrated_v2.RDS")
tes <- merge(tes_endt,y=c(tes_me,tes_immu,tes_epi),add.cell.ids=c("endt", "me","immu","epi"),merge.data = TRUE)
DefaultAssay(tes) <- "RNA"


# clustering
RNA.res = RNARunSeurat(inputMat = tes@assays$RNA, 
                       project = "LUSC_GSE178362", 
                       min.c = 10,
                       min.g = 500,
                       dims.use = 1:30,
                       variable.genes = 2000, 
                       organism = "GRCh38",
                       cluster.res = 1,
                       genes.test.use = "presto",
                       only.pos = TRUE,
                       genes.cutoff = 1e-05)
tes@meta.data
RNA.res$RNA= RNAAnnotateCelltype(RNA = RNA.res$RNA, 
                                genes = RNA.res$genes,
                                signatures = "human.immune.CIBERSORT",
                                min.score = 0.3)
RNA.res$RNA@meta.data = cbind(RNA.res$RNA@meta.data, tes@meta.data[colnames(RNA.res$RNA),, drop = FALSE])
RNA.res$RNA@meta.data$assign.CIBERSORT = RNA.res$RNA@meta.data$assign.ident
RNA.res$RNA@meta.data$assign.curated = as.character((RNA.res$RNA$seurat_clusters))
saveRDS(RNA.res, "D:/project/deconvolution_algorithm/coding/Spatial_mapping/CytoBulk/analysis/preprocessing/Seurat_lung.rds")

# if the sample is tumor sample, using SCEVAN to infer tumor cell.
results <- SCEVAN::pipelineCNA(RNA.res$RNA[["RNA"]]@counts,  par_cores = 1, SUBCLONES = TRUE, plotTree = TRUE)
results<-readRDS("D:/project/deconvolution_algorithm/coding/Spatial_mapping/data/cnv_lung.rds")

#write10xCounts(x = RNA.res@assays$RNA@counts, path = "D:/project/deconvolution_algorithm/coding/Spatial_mapping/CytoBulk/analysis/preprocessing/cnv/count.mtx")
#write.table (RNA.res$RNA[["RNA"]]@counts, file ="D:/project/deconvolution_algorithm/coding/Spatial_mapping/CytoBulk/analysis/preprocessing/cnv/count.txt", sep ="\t", row.names =TRUE, col.names =TRUE, quote =FALSE)
write.table (RNA.res$RNA@meta.data, file ="D:/project/deconvolution_algorithm/coding/Spatial_mapping/CytoBulk/analysis/preprocessing/cnv/cell.txt", sep ="\t", row.names =TRUE, col.names =TRUE, quote =FALSE)
#cnv_data = run_cnv("D:/project/deconvolution_algorithm/coding/Spatial_mapping/CytoBulk/analysis/preprocessing/cnv")


## visualization to curated cell type
## get the umap plot of the project
baseplot<-DimPlot(RNA.res$RNA)
baseplot + labs(title = "Clustering of 2,700 PBMCs")

DefaultAssay(RNA.res$RNA) = "RNA"
VisualizeUmap(SeuratObj = RNA.res$RNA, type = "RNA", genes = c("CTSW"), ncol = 1,
              width = 15, height = 4, name = paste0(RNA.res$RNA@project.name, "_CD8Tex"))

RNA.res= Level1_RNAAnnotateCelltype(RNA.res,"D:/project/deconvolution_algorithm/coding/Spatial_mapping/CytoBulk/analysis/preprocessing/")
RNA.res = Level3_RNAAnnotateCelltype(RNA.res,"D:/project/deconvolution_algorithm/coding/Spatial_mapping/CytoBulk/analysis/preprocessing/")

annotations_file=system.file("extdata", "oligodendroglioma_annotations_downsampled.txt", package = "infercnv")