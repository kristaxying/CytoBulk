
library(Giotto)
library(scran)
library(sva)

run_giotto <- function(sc_data, sc_anno, python_path, out_dir, project, save=TRUE){
  my_instructions = createGiottoInstructions(python_path = python_path)
  sc_giotto_object <- createGiottoObject(raw_exprs = sc_data,
                                        instructions = my_instructions,
                                        cell_metadata = sc_anno)
  sc_giotto_object <- filterGiotto(gobject = sc_giotto_object, 
                                  expression_threshold =1, #1
                                  gene_det_in_min_cells = 10, #10
                                  min_det_genes_per_cell = 5) #5

  sc_giotto_object <- normalizeGiotto(gobject = sc_giotto_object)
  sc_giotto_object <- calculateHVG(gobject = sc_giotto_object,save_plot=FALSE,return_plot=FALSE,show_plot=FALSE)

  gene_metadata = fDataDT(sc_giotto_object)
  featgenes = gene_metadata[hvg == 'yes']$gene_ID
  sc_giotto_object <- runPCA(gobject = sc_giotto_object, genes_to_use = featgenes, scale_unit = F)
  sc_giotto_object <- createNearestNetwork(gobject = sc_giotto_object, dimensions_to_use = 1:10, k = 10)
  sc_giotto_object <- doLeidenCluster(gobject = sc_giotto_object, resolution = 0.4, n_iterations = 1000)
  cell_metadata = pDataDT(sc_giotto_object)
  
  scran_markers_subclusters = findMarkers_one_vs_all(gobject = sc_giotto_object,
                                                    method = 'scran',
                                                    expression_values = 'normalized',
                                                    cluster_column = "leiden_clus")
  id<-sc_giotto_object@cell_metadata$curated_cell_type
  if(length(id)>30){
      Sig_scran <- unique(scran_markers_subclusters$genes[which(scran_markers_subclusters$ranking <= 150)])
  }else{
      Sig_scran <- unique(scran_markers_subclusters$genes[which(scran_markers_subclusters$ranking <= 150)])
  }
  norm_exp<-2^(sc_giotto_object@norm_expr)-1


  ExprSubset<-norm_exp[Sig_scran,]
  Sig_exp<-NULL
  for (i in unique(id)){
    Sig_exp<-cbind(Sig_exp,(apply(ExprSubset,1,function(y) mean(y[which(id==i)]))))
  }
  colnames(Sig_exp)<-unique(id)

  if(save==TRUE){
      write.table(Sig_exp,file = paste0(out_dir,"/",project,"_marker.txt"),row.names = TRUE,sep="\t",col.names=TRUE,quote =FALSE)
  }

}

run_combat <- function(bulk, meta,out_dir='./', project='',save=TRUE){
  #print(setdiff(colnames(bulk), rownames(meta)))

  row.names(meta)<-meta$cells
  meta$cells<-NULL
  combat_edata = ComBat(dat=bulk, batch=meta$batch, mod=NULL, par.prior=TRUE)
  
  if(save==TRUE){
      write.table(combat_edata,file = paste0(out_dir,"/",project,"_batch_effected.txt"),row.names = TRUE,sep="\t",col.names=TRUE,quote =FALSE)
  }
}