library(BayesPrism)
library(optparse)
library(Seurat)
library(hdf5r)
library(SCDC)

read_rna<-function(opt){
  opt$bulk_rna <- read.csv(opt$bulk_rna,row.names = 1,sep='\t')
  opt$sc_data <- read.csv(opt$sc_fn,row.names = 1,sep='\t')
  opt$sc_meta <- read.csv(opt$sc_meta,row.names = 1,sep='\t')
  opt$cell_type <-opt$sc_meta$curated_cell_type
  opt$cell_state <- opt$sc_meta$curated_cell_type
  return(opt)
}

run_BayesPrism <- function(opt){
  sc.dat.filtered <- cleanup.genes (input=opt$sc_data,
                                    input.type="count.matrix",
                                    species="hs", 
                                    gene.group=c( "Rb","Mrp","other_Rb","chrM","MALAT1","chrX","chrY") ,
                                    exp.cells=5)
  sc.dat.filtered.pc <-  select.gene.type (sc.dat.filtered,
                                           gene.type = "protein_coding")
  diff.exp.stat <- get.exp.stat(sc.dat=opt$sc_data[,colSums(opt$sc_data>0)>3],# filter genes to reduce memory use
                                cell.type.labels=opt$cell_type,
                                cell.state.labels=opt$cell_state,
                                pseudo.count=0.1, #a numeric value used for log2 transformation. =0.1 for 10x data, =10 for smart-seq. Default=0.1.
                                cell.count.cutoff=50, # a numeric value to exclude cell state with number of cells fewer than this value for t test. Default=50.
                                n.cores=1)#number of threads
  sc.dat.filtered.pc.sig <- select.marker (sc.dat=sc.dat.filtered.pc,
                                           stat=diff.exp.stat,
                                           pval.max=0.01,
                                           lfc.min=0.1)
  myPrism <- new.prism(
    reference=sc.dat.filtered.pc, 
    mixture=opt$bulk_rna,
    input.type="count.matrix", 
    cell.type.labels = opt$cell_type, 
    cell.state.labels = opt$cell_state,
    key=NULL,
    outlier.cut=0.01,
    outlier.fraction=0.1,
  )
  bp.res <- run.prism(prism = myPrism, n.cores=50)
  theta <- get.fraction (bp=bp.res,
                         which.theta="final",
                         state.or.type="type")
  return(theta)
}

run_scdc <- function(opt){
  sc_data = t(opt$sc_data)
  fdata <- rownames(sc_data)
  pdata <- opt$sc_meta
  sc_set <- getESET(sc_data, fdata = fdata, pdata = pdata)
  cell_type = c(unique(opt$sc_meta$curated_cell_type))
  
  seger.qc <- SCDC_qc_ONE(sc_set, ct.varname = "curated_cell_type", sample = "donor_id", scsetname = "Single Cell",,
                      ct.sub = cell_type, qcthreshold = 0.5, generate.figure=FALSE)
  fdata <- rownames(t(opt$bulk_rna))
  pdata <- cbind(cellname = colnames(t(opt$bulk_rna)), subjects = nrow("patient35"))
  bulk_set <- getESET(t(opt$bulk_rna), fdata = fdata, pdata = pdata)
  ens <- SCDC_prop_ONE(bulk.eset = bulk_set, sc.eset = seger.qc, ct.varname = "curated_cell_type",
                      sample = "sample", truep = NULL, ct.sub =  unique(opt$sc_meta$curated_cell_type), search.length = 0.01, grid.search = T)
  print(ens)
  return(ens$w_table)
}



run <- function(opt){
  opt <- read_rna(opt)
  
  if(opt$method=="BayesPrism"){
    BayesPrism_results <- run_BayesPrism(opt)
    write.csv(BayesPrism_results,file = paste0(opt$out_dir,"/",opt$project,"_BayesPrism_data.csv"),row.names = TRUE)
  }
  if(opt$method=="SCDC"){
    scdc_results <- run_scdc(opt)
    write.csv(scdc_results,file = paste0(opt$out_dir,"/",opt$project,"_scdc_data.csv"),row.names = TRUE)
  }
}
option_list = list(
  make_option("--bulk_rna", type="character", default=NULL,
              help="The path of st expression data", metavar="character"),
  make_option("--sc_meta", type="character", default=NULL,
              help="The path of st expression data", metavar="character"),
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