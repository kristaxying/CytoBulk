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
  if(opt$project=="MM_GSE151310"){
      sc_data <- subset(x = sc_data, subset = (Celltype..minor.lineage. %in% c("B","Th1","Th17","CD8Tcm","MAIT","CD4Tn","CD8Teff","CD8Tex","CD8Tem","cDC1","cDC2","CD8Tn","M1","M2","Mast","Monocyte","NK","pDC","CD4Tconv","Plasma","Tprolif","Treg","Th2")))}else{
      sc_data <- subset(x = sc_data, subset = (Celltype..minor.lineage. %in% c("B","Th1","Th17","CD8Tcm","MAIT","CD4Tn","CD8Teff","CD8Tex","CD8Tem","cDC1","cDC2","CD8Tn","M1","M2","Mast","Monocyte","NK","pDC","CD4Tconv","Plasma","Tprolif","Treg","Tfh","Th2","CD4Teff","Fibroblasts","Epithelial","Endothelial","Myofibroblasts")))}
  
  if(opt$project=="HNSC_GSE139324"){
      sc_data <- subset(x = sc_data, subset = (Disease %in% c("Healthy")))
      if(opt$method=="rctd"){
          sc_data <- subset(x = sc_data, subset = (Celltype..minor.lineage. %in% c("B","Tfh","Th1","Th17","CD8Tcm","CD4Tn","CD8Teff","CD8Tex","CD8Tem","cDC1","cDC2","CD8Tn","M1","MAIT","Monocyte","NK","pDC","CD4Tconv","Plasma","Tprolif","Treg","Th2")))}}

  opt$sc_data <- GetAssay(sc_data,assay = "RNA")@counts
  opt$sc_anno <- sc_data@meta.data[,"Celltype..minor.lineage."]
  opt$sc_meta <- sc_data@meta.data
  opt$st_meta <- read.csv(opt$meta_fn,row.names = 1)
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
    ct.select = unique(opt$sc_meta$Celltype..minor.lineage.),
    sample.varname = "Sample",
    minCountGene = 100,
    minCountSpot = 5)
  CARD_obj = CARD_deconvolution(CARD_object = CARD_obj)
  print(CARD_obj@algorithm_matrix)
  write.csv(CARD_obj@algorithm_matrix$B,file = paste0(opt$out_dir,"/",opt$project,"_card_B.csv"),row.names = TRUE)
  return(CARD_obj@Proportion_CARD)
}
run_rctd <- function(opt){
  card_ref = 2^(as.matrix(opt$sc_data))
  card_ref = round(card_ref)
  rctd_st = 2^(as.matrix(opt$st_data))
  rctd_st = round(rctd_st)
  meta_data <- opt$sc_meta
  cell_types <- meta_data$Celltype..minor.lineage.
  names(cell_types) <- row.names(meta_data)
  cell_types <- as.factor(cell_types)
  reference <- Reference(card_ref, cell_types)
  puck <- SpatialRNA(opt$st_meta[,c("x","y")], rctd_st)
  myRCTD <- create.RCTD(puck, reference, max_cores = 2)
  myRCTD <- run.RCTD(myRCTD, doublet_mode = 'doublet')
  results <- myRCTD@results
  norm_weights = normalize_weights(results$weights) 
  cell_type_names <- myRCTD@cell_type_info$info[[2]]
  return(list("weight"=norm_weights,"cell_types"=cell_type_names))
}

run_spatialdwls <- function(opt){
  my_instructions = createGiottoInstructions(python_path = '/home/wangxueying/miniconda3/envs/cytobulk/bin/python')
  st_giotto_object = createGiottoObject(raw_exprs = opt$st_data,
                                        spatial_locs = opt$st_meta,
                                        instructions = my_instructions)
  st_giotto_object <- filterGiotto(gobject = st_giotto_object, 
                                   expression_threshold =0.5, 
                                   gene_det_in_min_cells = 10, 
                                   min_det_genes_per_cell = 0)
  st_giotto_object <- normalizeGiotto(gobject = st_giotto_object)
  st_giotto_object <- calculateHVG(gobject = st_giotto_object)
  gene_metadata = fDataDT(st_giotto_object)
  featgenes = gene_metadata[hvg == 'yes']$gene_ID
  st_giotto_object <- runPCA(gobject = st_giotto_object, genes_to_use = featgenes, scale_unit = F)
  signPCA(st_giotto_object, genes_to_use = featgenes, scale_unit = F)
  st_giotto_object <- createNearestNetwork(gobject = st_giotto_object, dimensions_to_use = 1:10, k = 10)
  st_giotto_object <- doLeidenCluster(gobject = st_giotto_object, resolution = 0.4, n_iterations = 1000)
  
  
  
  sc_giotto_object <- createGiottoObject(raw_exprs = opt$sc_data,
                                         instructions = my_instructions,
                                         cell_metadata = opt$sc_anno)
  sc_giotto_object <- filterGiotto(gobject = sc_giotto_object, 
                                   expression_threshold =0.5, 
                                   gene_det_in_min_cells = 10, 
                                   min_det_genes_per_cell = 0)
  sc_giotto_object <- normalizeGiotto(gobject = sc_giotto_object)
  sc_giotto_object <- calculateHVG(gobject = sc_giotto_object)
  gene_metadata = fDataDT(sc_giotto_object)
  featgenes = gene_metadata[hvg == 'yes']$gene_ID
  sc_giotto_object <- runPCA(gobject = sc_giotto_object, genes_to_use = featgenes, scale_unit = F)
  signPCA(sc_giotto_object, genes_to_use = featgenes, scale_unit = F)
  sc_giotto_object <- createNearestNetwork(gobject = sc_giotto_object, dimensions_to_use = 1:10, k = 10)
  sc_giotto_object <- doLeidenCluster(gobject = sc_giotto_object, resolution = 0.4, n_iterations = 1000)
  cell_metadata = pDataDT(sc_giotto_object)
  print(cell_metadata)
  scran_markers_subclusters = findMarkers_one_vs_all(gobject = sc_giotto_object,
                                                   method = 'scran',
                                                   expression_values = 'normalized',
                                                   cluster_column = "leiden_clus")

  Sig_scran <- unique(scran_markers_subclusters$genes[which(scran_markers_subclusters$ranking <= 100)])
  norm_exp<-2^(sc_giotto_object@norm_expr)-1
    id<-sc_giotto_object@cell_metadata$V1
    ExprSubset<-norm_exp[Sig_scran,]
    Sig_exp<-NULL
    for (i in unique(id)){
      Sig_exp<-cbind(Sig_exp,(apply(ExprSubset,1,function(y) mean(y[which(id==i)]))))
    }
  colnames(Sig_exp)<-unique(id)
  
  write.table(Sig_exp,file = paste0(opt$out_dir,"/",opt$project,"_marker.csv"),row.names = TRUE,sep=",",col.names=TRUE,quote =FALSE)
  
  
  st_giotto_object = runDWLSDeconv(gobject = st_giotto_object, sign_matrix = Sig_exp,cluster_column = "leiden_clus")
  st_giotto_result = as.data.frame(st_giotto_object@spatial_enrichment$DWLS)[-1]
  giotto_sc = cell_metadata
  return(list("data"=st_giotto_result,"meta"=giotto_sc))
}


run <- function(opt){
  print("read spatial expression data")

  opt <- read_rna(opt)

  if(opt$method=="spotlight"){
    spotlight_input <- spotlight_preprocess(opt)
    spotlight_results <- run_spotlight(spotlight_input)
    write.csv(spotlight_results,file = paste0(opt$out_dir,"/",opt$project,"_spotlight_data.csv"),row.names = TRUE)
  }
  if(opt$method=="card"){
    card_results <- run_card(opt)
    write.csv(card_results,file = paste0(opt$out_dir,"/",opt$project,"_card_data.csv"),row.names = TRUE)
  }
  if(opt$method=="rctd"){
    rctd_results <- run_rctd(opt)
    write.csv(rctd_results$weight,file = paste0(opt$out_dir,"/",opt$project,"_rctd_data.csv"),row.names = TRUE)
  }
 if(opt$method=="spatialdwls"){
   spatialdwls_results <- run_spatialdwls(opt)
   write.csv(spatialdwls_results$data,file = paste0(opt$out_dir,"/",opt$project,"_spatialdwls_data.csv"),row.names = TRUE)
   write.csv(spatialdwls_results$meta,file = paste0(opt$out_dir,"/",opt$project,"_spatialdwls_meta.csv"),row.names = TRUE)
 }
  if(opt$method=="read_data"){
   #write.table(opt$sc_data,file = paste0(opt$out_dir,"/",opt$project,"_sc_data.txt"),row.names = TRUE,sep='\t',quote =FALSE)
   samples = colnames(opt$sc_data)
   print(samples)
   row.names(opt$sc_anno) <- samples
   print(opt$sc_anno)
   #row.names(opt$sc_meta) <- samples
   #write.table(t(opt$sc_data),file = paste0(opt$out_dir,"/",opt$project,"_sc_data_cell2.txt"),row.names = TRUE,sep='\t',quote =FALSE)
   write.table(opt$sc_anno,file = paste0(opt$out_dir,"/",opt$project,"_sc_cell.txt"),row.names = TRUE,sep='\t',col.names=FALSE,quote =FALSE)
   #write.table(opt$sc_meta,file = paste0(opt$out_dir,"/",opt$project,"_sc_meta.txt"),row.names = TRUE,sep='\t',quote =FALSE)
   #write.table(opt$st_data,file = paste0(opt$out_dir,"/",opt$project,"_st_data.txt"),row.names = TRUE,sep='\t',quote =FALSE)
   
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