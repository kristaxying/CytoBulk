## install packages
library(MAESTRO)
library(Seurat)
library(ggplot2)
library(presto)
library(Gmisc)
library(SCEVAN)


source("D:/project/deconvolution_algorithm/coding/Spatial_mapping/CytoBulk/analysis/preprocessing/sc_utils.R")

# for multiple samples
tes_endt<-readRDS("C:/Users/xywang85/Downloads/GSE178360_endo.sub_SAE10x3.integrated_temp_v2 (1).RDS/GSE178360_endo.sub_SAE10x3.integrated_temp_v2.RDS")
tes_me<-readRDS("C:/Users/xywang85/Downloads/GSE178360_mes.sub_SAE10x3.integrated_v2.RDS/GSE178360_mes.sub_SAE10x3.integrated_v2.RDS")
tes_immu<-readRDS("C:/Users/xywang85/Downloads/GSE178360_immune_platelet.sub_SAE10x3.integrated_temp.RDS/GSE178360_immune_platelet.sub_SAE10x3.integrated_temp.RDS")
tes_epi<-readRDS("C:/Users/xywang85/Downloads/GSE178360_epi.sub_SAE10x3.integrated_v2.RDS/GSE178360_epi.sub_SAE10x3.integrated_v2.RDS")
tes <- merge(tes_endt,y=c(tes_me,tes_immu,tes_epi),add.cell.ids=c("endt", "me","immu","epi"),merge.data = TRUE)
DefaultAssay(tes) <- "RNA"


# clustering
RNA.res = RNARunSeurat(inputMat = tes@assays$RNA, 
                       project = "GSE178360", 
                       min.c = 10,
                       min.g = 500,
                       dims.use = 1:30,
                       variable.genes = 2000, 
                       organism = "GRCh38",
                       cluster.res = 1,
                       genes.test.use = "presto",
                       only.pos = TRUE,
                       genes.cutoff = 1e-05)
# cell annotation
RNA.res$RNA= RNAAnnotateCelltype(RNA = RNA.res$RNA, 
                                genes = RNA.res$genes,
                                signatures = "human.immune.CIBERSORT",
                                min.score = 0.5)

meta<-NULL
meta = read.csv('D:/project/deconvolution_algorithm/coding/Spatial_mapping/CytoBulk/analysis/preprocessing/cnv/cell_final.txt',sep='\t')
rownames(meta) <- meta$X
meta$X <- NULL
collist <- colnames(RNA.res$RNA@meta.data)
for(i in collist){
  RNA.res$RNA@meta.data$i <- NULL
}
RNA.res$RNA@meta.data <- RNA.res$RNA@meta.data[ , -which(colnames(RNA.res$RNA@meta.data) %in% collist)]
RNA.res$RNA@meta.data$assign.CIBERSORT = RNA.res$RNA@meta.data$assign.ident
RNA.res$RNA@meta.data$assign.curated = as.character((RNA.res$RNA$seurat_clusters))
RNA.res$RNA@meta.data <- meta
# add cell type annotation from orginal study
RNA.res$RNA@meta.data = cbind(RNA.res$RNA@meta.data, meta[colnames(RNA.res$RNA),, drop = FALSE])
RNA.res$RNA@meta.data = cbind(RNA.res$RNA@meta.data, tes@meta.data[colnames(RNA.res$RNA),, drop = FALSE])
test2<- RNA.res$RNA
test2@meta.data
#saveRDS(RNA.res, "D:/project/deconvolution_algorithm/coding/Spatial_mapping/CytoBulk/analysis/preprocessing/Seurat_lung.rds")

# using visualizations to curate cell type

#write10xCounts(x = RNA.res@assays$RNA@counts, path = "D:/project/deconvolution_algorithm/coding/Spatial_mapping/CytoBulk/analysis/preprocessing/cnv/count.mtx")
#write.table (RNA.res$RNA[["RNA"]]@counts, file ="D:/project/deconvolution_algorithm/coding/Spatial_mapping/CytoBulk/analysis/preprocessing/cnv/count.txt", sep ="\t", row.names =TRUE, col.names =TRUE, quote =FALSE)
write.table (RNA.res$RNA@meta.data, file ="D:/project/deconvolution_algorithm/coding/Spatial_mapping/CytoBulk/analysis/preprocessing/cnv/cell.txt", sep ="\t", row.names =TRUE, col.names =TRUE, quote =FALSE)
#cnv_data = run_cnv("D:/project/deconvolution_algorithm/coding/Spatial_mapping/CytoBulk/analysis/preprocessing/cnv")
RNA.res$RNA@meta.data$assign.level3_anno<-NULL
RNA.res$RNA@meta.data$assign.level1_anno<-NULL

# visualization to curated cell type

## get the umap plot of the project
baseplot<-DimPlot(RNA.res$RNA,reduction = "umap",group.by="cell_type",label=TRUE)
baseplot + labs(title = "Clustering of GSE178360")

## for specific marker genes of CD4Teff
plot <- FeaturePlot(RNA.res$RNA, features = "CD8A")
HoverLocator(plot = plot, information = FetchData(RNA.res$RNA, vars = c("cell_type", "seurat_clusters","assign.CIBERSORT")))
VisualizeUmap(SeuratObj = RNA.res$RNA, type = "RNA", genes = c("CTSW","GNLY"), ncol = 1,
              width = 15, height = 4, name = paste0(RNA.res$RNA@project.name, "_CD4Teff"))
HoverLocator(plot = plot, information = FetchData(pbmc3k.final, vars = c("ident", "PC_1", "nFeature_RNA")))
RNA.res= Level1_RNAAnnotateCelltype(RNA.res,"D:/project/deconvolution_algorithm/coding/Spatial_mapping/CytoBulk/analysis/preprocessing/")
RNA.res = Level3_RNAAnnotateCelltype(RNA.res,"D:/project/deconvolution_algorithm/coding/Spatial_mapping/CytoBulk/analysis/preprocessing/")

annotations_file=system.file("extdata", "oligodendroglioma_annotations_downsampled.txt", package = "infercnv")