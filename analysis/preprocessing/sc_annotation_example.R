## install packages
library(MAESTRO)
library(Seurat)
library(ggplot2)
library(presto)
library(Gmisc)
library(SCEVAN)


source("D:/project/deconvolution_algorithm/coding/Spatial_mapping/CytoBulk/analysis/preprocessing/sc_utils.R")
setwd("D:/project/pathway/spatalk_plus/analysis/data/ER/CID4290") 

## read count table and metadata
new_counts <- read.csv(file="./sc/scexp.csv",header = TRUE)
rownames(new_counts) <- new_counts$GENES
new_counts$GENES <- NULL
meta = read.csv('./sc/metadata_ori.csv')
rownames(meta) <- meta$X
meta$X <- NULL
mydata <- CreateSeuratObject(counts = new_counts, min.cells = 3, project = "CID4290")

# clustering
RNA.res = RNARunSeurat(inputMat = mydata@assays$RNA, 
                       project = "CID4290", 
                       min.c = 10,
                       min.g = 500,
                       dims.use = 1:30,
                       variable.genes = 2000, 
                       organism = "GRCh38",
                       cluster.res = 1,
                       genes.test.use = "presto",
                       only.pos = TRUE,
                       genes.cutoff = 1e-05)
# cell annotation using marker genes
RNA.res$RNA= RNAAnnotateCelltype(RNA = RNA.res$RNA, 
                                genes = RNA.res$genes,
                                signatures = "human.immune.CIBERSORT",
                                min.score = 0.5)
# add metadata
RNA.res$RNA@meta.data$assign.CIBERSORT = RNA.res$RNA@meta.data$assign.ident
RNA.res$RNA@meta.data$assign.curated = as.character((RNA.res$RNA$seurat_clusters))
RNA.res$RNA@meta.data = cbind(RNA.res$RNA@meta.data, meta[colnames(RNA.res$RNA),, drop = FALSE])

# write annotated information to curate
write.table (RNA.res$RNA@meta.data, file ="./cell_new.txt", sep ="\t", row.names =TRUE, col.names =TRUE, quote =FALSE)
write.table (cluster.genes, file ="./marker.txt", sep ="\t", row.names =TRUE, col.names =TRUE, quote =FALSE)
# read curated data
meta = read.csv('./cell_new.txt',sep='\t')
rownames(meta) <- meta$X
meta$X <- NULL

# delete original annotation
collist <- colnames(RNA.res$RNA@meta.data)
RNA.res$RNA@meta.data <- RNA.res$RNA@meta.data[ , -which(colnames(RNA.res$RNA@meta.data) %in% collist)]
RNA.res$RNA@meta.data = cbind(RNA.res$RNA@meta.data, meta[colnames(RNA.res$RNA),, drop = FALSE])
Idents(object = RNA.res$RNA) <- 'seurat_clusters'
Idents(object = RNA.res$RNA)
# find marker gene
cluster.genes <- FindAllMarkersMAESTRO(object = RNA.res$RNA, min.pct = 0.1, logfc.threshold = 0.25, test.use = "presto", only.pos = FALSE)
RNA.res$genes <- cluster.genes

RNA.res <- SetAllIdent(object = RNA.res, id = 'seurat_clusters')
# detailed annotation
RNA.res = Level3_RNAAnnotateCelltype(RNA.res,"D:/project/deconvolution_algorithm/coding/Spatial_mapping/CytoBulk/analysis/preprocessing/")
RNA.res= Level1_RNAAnnotateCelltype(RNA.res,"D:/project/deconvolution_algorithm/coding/Spatial_mapping/CytoBulk/analysis/preprocessing/")

# plot
Idents(object = RNA.res$RNA) <- 'assign.level3_anno'
DimPlot(RNA.res$RNA,reduction = "umap")
RNA.res$genes








