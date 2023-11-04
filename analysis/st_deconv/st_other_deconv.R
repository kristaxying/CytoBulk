library(optparse)
library(ggplot2)
library(SPOTlight)
library(SingleCellExperiment)
library(SpatialExperiment)
library(scater)
library(scran)
library(Matrix)
library(SeuratDisk)
library(Seurat)
library(Giotto)
library(CARD)
library(spacexr)
library(Biobase)
## install the packages
#if (!require("BiocManager", quietly = TRUE))
#install.packages("BiocManager")
#BiocManager::install("GENIE3")
#install.packages("ppcor")


## execute each line in if(F)
if(F){
  opt <- list()
  opt$rna_fn <- 'E:/CytoBulk/pretrain_evaluation/OV_GSE154763/training_data/SCC_GSE145328_expression_test.csv'
  opt$sc_fn  <- 'E:/CytoBulk/pretrain_evaluation/OV_GSE154763/OV_GSE154763.h5Seurat'
  opt$out_dir <- 'D:/project/casuality/casuality/out/'
  opt$project <- 'NSCLC_GSE153935'
  opt$st_meta <- 'E:/CytoBulk/pretrain_evaluation/SCC_GSE145328/training_data/st_meta_test.csv'
}
if(F){
  opt <- list()
  opt$rna_fn <- '/data1/wangxueying/cytobulk/eval_data/THCA_GSE148673_expression_test.csv'
  opt$sc_fn  <- '/data1/wangxueying/cytobulk/eval_data/THCA_GSE148673.h5Seurat'
  opt$meta <-
  opt$out_dir <- '/data1/wangxueying/cytobulk/out/'
  opt$project <- 'SCC_GSE145328'
}

read_rna <- function(opt){
  opt$st_data <- t(read.csv(opt$rna_fn,row.names = 1))
  opt$sparse_st_data <- Matrix(as.matrix(opt$st_data), sparse=T)
  sc_data <- LoadH5Seurat(opt$sc_fn,assays = "RNA")
  opt$sc_data <- GetAssay(sc_data,assay = "RNA")@counts
  opt$sc_anno <- sc_data@meta.data[,"Celltype..minor.lineage."]
  opt$sc_meta <- sc_data@meta.data
  opt$st_meta <- read.csv(opt$st_meta,row.names = 1)
  return(opt)
}

spotlight_preprocess <- function(opt){
  card_sc <- opt$sc_data
  dec <- modelGeneVar(card_sc)
  hvg <- getTopHVGs(dec, n = 3000)
  genes <- !grepl(pattern = "^Rp[l|s]|Mt", x = rownames(card_sc))
  mgs <- scoreMarkers(card_sc,groups =opt$sc_anno,subset.row = genes)
  mgs_fil <- lapply(names(mgs), function(i) {
    x <- mgs[[i]]
    # Filter and keep relevant marker genes, those with AUC > 0.8
    x <- x[x$mean.AUC > 0.7, ]
    # Sort the genes from highest to lowest weight
    x <- x[order(x$mean.AUC, decreasing = TRUE), ]
    # Add gene and cluster id to the dataframe
    x$gene <- rownames(x)
    x$cluster <- i
    data.frame(x)
  })
  mgs_df <- do.call(rbind, mgs_fil)
  return_data <- list("sce" = card_sc, "ste" = opt$sparse_st_data, "anno" = opt$sc_anno, "mgs" = mgs_df, "hvg" = hvg)
  return(return_data)
}

run_spotlight <- function(spotlight){
  res <- SPOTlight(
    x = spotlight$sce,
    y = spotlight$ste,
    groups = spotlight$anno,
    mgs = spotlight$mgs,
    hvg = spotlight$hvg,
    weight_id = "mean.AUC",
    group_id = "cluster",
    gene_id = "gene")
  return(res$mat)
}


run_card <- function(opt){
  CARD_obj = createCARDObject(
    sc_count = opt$sc_data,
    sc_meta = opt$sc_meta,
    spatial_count = opt$sparse_st_data,
    spatial_location = opt$st_meta,
    ct.varname = "Celltype..minor.lineage.",
    ct.select = unique(sc_meta$Celltype..minor.lineage.),
    sample.varname = "Sample",
    minCountGene = 100,
    minCountSpot = 5)
  CARD_obj = CARD_deconvolution(CARD_object = CARD_obj)
  return(CARD_obj@Proportion_CARD)
}
run_rctd <- function(opt){
  meta_data <- opt$sc_meta
  cell_types <- meta_data$Celltype..minor.lineage.
  names(cell_types) <- row.names(meta_data)
  cell_types <- as.factor(cell_types)
  reference <- Reference(as.matrix(opt$sc_data), cell_types)
  puck <- SpatialRNA(opt$st_meta[,c("x","y")], opt$st_data)
  myRCTD <- create.RCTD(puck, reference, max_cores = 2)
  myRCTD <- run.RCTD(myRCTD, doublet_mode = 'doublet')
  results <- myRCTD@results
  norm_weights = normalize_weights(results$weights) 
  cell_type_names <- myRCTD@cell_type_info$info[[2]]
  return(list("weight"=norm_weights,"cell_types"=cell_type_names))
}

run_spatialdwls <- function(opt){
  my_instructions = createGiottoInstructions(python_path = 'D:/anaconda/ana/envs/py10/python.exe')
  st_giotto_object = createGiottoObject(raw_exprs = opt$st_data,
                                        spatial_locs = opt$st_meta,
                                        instructions = my_instructions)
  st_giotto_object <- filterGiotto(gobject = st_giotto_object, 
                                   expression_threshold =0.5, 
                                   gene_det_in_min_cells = 10, 
                                   min_det_genes_per_cell = 0)
  st_giotto_object <- normalizeGiotto(gobject = st_giotto_object)
  sc_giotto_object <- createGiottoObject(raw_exprs = opt$sc_data,
                                         instructions = my_instructions)
  sc_giotto_object <- filterGiotto(gobject = sc_giotto_object, 
                                   expression_threshold =0.5, 
                                   gene_det_in_min_cells = 10, 
                                   min_det_genes_per_cell = 0)
  sc_giotto_object <- normalizeGiotto(gobject = sc_giotto_object)
  sc_giotto_object <- runPCA(gobject = sc_giotto_object)
  sc_giotto_object <- runUMAP(sc_giotto_object, dimensions_to_use = 1:5)
  sc_giotto_object = doKmeans(sc_giotto_object, centers = 4, name = 'kmeans_clus')
  cluster_similarities = getClusterSimilarity(sc_giotto_object,
                                              cluster_column = 'kmeans_clus')
  mini_giotto_single_cell = mergeClusters(sc_giotto_object, 
                                          cluster_column = 'kmeans_clus', 
                                          min_cor_score = 0.7, 
                                          force_min_group_size = 4)
  gini_markers = findGiniMarkers_one_vs_all(gobject = mini_giotto_single_cell,
                                            cluster_column = 'kmeans_clus')
  
  

  
  st_giotto_object = doKmeans(st_giotto_object, centers = 4, name = 'kmeans_clus')
  cluster_similarities = getClusterSimilarity(st_giotto_object,
                                              cluster_column = 'kmeans_clus')
  mini_st_giotto_object = mergeClusters(st_giotto_object, 
                                          cluster_column = 'kmeans_clus', 
                                          min_cor_score = 0.7, 
                                          force_min_group_size = 4)
  splits = getDendrogramSplits(mini_st_giotto_object, cluster_column = 'merged_cluster')
  splits = getDendrogramSplits(st_giotto_object, cluster_column = 'merged_cluster')
  
  gini_markers = findGiniMarkers_one_vs_all(gobject = mini_giotto_single_cell,
                                            cluster_column = 'leiden_clus')
  rank_matrix = makeSignMatrixRank(sc_matrix = as.matrix(opt$sc_data), sc_cluster_ids = as.vector(sc_meta$Celltype..minor.lineage.))
  st_giotto_object = runHyperGeometricEnrich(gobject = st_giotto_object,
                                             sign_matrix = rank_matrix)
  spatCellPlot(gobject = st_giotto_object, 
               spat_enr_names = 'hypergeometric',
               cell_annotation_values = as.vector(sc_meta$Celltype..minor.lineage.),
               cow_n_col = 2,coord_fix_ratio = NULL, point_size = 2.75)

  st_giotto_object = runDWLSDeconv(gobject = st_giotto_object, sign_matrix = rank_matrix)
  return(st_giotto_object)
}


run <- function(opt){
  print("read spatial expression data")
  print(opt)
  opt <- read_rna(opt)
  if(opt$method=="spotlight"){
    spotlight_input <- spotlight_preprocess(opt)
    spotlight_results <- run_spotlight(spotlight_input)
    write.csv(spotlight_results,file = paste0(opt$out_dir,"/",opt$project,"spotlight_data.csv"),row.names = TRUE)
  }
  if(opt$method=="card"){
    card_results <- run_card(opt)
    write.csv(card_results,file = paste0(opt$out_dir,"/",opt$project,"card_data.csv"),row.names = TRUE)
  }
  if(opt$method=="rctd"){
    rctd_results <- run_rctd(opt)
    write.csv(rctd_results$weight,file = paste0(opt$out_dir,"/",opt$project,"rctd_results_data.csv"),row.names = TRUE)
    write.csv(rctd_results$cell_types,file = paste0(opt$out_dir,"/",opt$project,"rctd_results_celltype.csv"),row.names = TRUE)
  }
 if(opt$method=="spatialdwls"){
   spatialdwls_results <- run_spatialdwls(opt)
   saveRDS(spatialdwls_results, file = paste0(opt$out_dir,"/",opt$project,"spatialdwls_results.rds"))
   
 }
}

## ignore
## library("optparse")
option_list = list(
  make_option("--rna_fn", type="character", default=NULL,
              help="The path of st expression data", metavar="character"),
  make_option("--st_meta", type="character", default=NULL,
              help="The path of st expression data", metavar="character"),
  make_option(c("-m","--meta_fn"), type="character", default=NULL,
     help="The path of st meta data", metavar="character"),
  make_option("--sc_fn", type="character", default=NULL,
              help="The path of sc h5seurat", metavar="character"),
  make_option("--project", type="character", default=NULL,
              help="The path of project name.", metavar="character"),
  make_option("--method", type="character", default=NULL,
              help="The path of project name.", metavar="character"),
  make_option("--out_dir", type="character", default='./',
              help="The path of output directory. [default=./]", metavar="character")
);

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

run(opt)