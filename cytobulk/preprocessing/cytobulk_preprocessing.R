library(sva)
library(scran)
library(Giotto)


run_giotto <- function(sc_data, sc_anno, python_path, out_dir, project, save=TRUE){

  my_instructions = createGiottoInstructions(python_path = python_path)
  sc_giotto_object <- createGiottoObject(raw_exprs = sc_data,
                                        instructions = my_instructions,
                                        cell_metadata = sc_anno)
  sc_giotto_object <- filterGiotto(gobject = sc_giotto_object, 
                                  expression_threshold =1, #1
                                  feat_det_in_min_cells = 10, #10
                                  min_det_feats_per_cell = 5) #5

  sc_giotto_object <- normalizeGiotto(gobject = sc_giotto_object)
  sc_giotto_object <- calculateHVF(gobject = sc_giotto_object,save_plot=FALSE,return_plot=FALSE,show_plot=FALSE)

  gene_metadata = fDataDT(sc_giotto_object)
  featgenes = gene_metadata[hvf == 'yes']$gene_ID
  sc_giotto_object <- runPCA(gobject = sc_giotto_object, genes_to_use = featgenes, scale_unit = F)
  sc_giotto_object <- createNearestNetwork(gobject = sc_giotto_object, dimensions_to_use = 1:10, k = 10)
  sc_giotto_object <- doLeidenCluster(gobject = sc_giotto_object, resolution = 0.4, n_iterations = 1000)
  cell_metadata = pDataDT(sc_giotto_object)
  
  scran_markers_subclusters = findMarkers_one_vs_all(gobject = sc_giotto_object,
                                                    method = 'scran',
                                                    expression_values = 'normalized',
                                                    cluster_column = "leiden_clus")
  id<-cell_metadata$curated_cell_type
  print(id)

  
  Sig_scran <- unique(scran_markers_subclusters$feats[which(scran_markers_subclusters$ranking <= 150)])
  norm_exp<-get_expression_values(sc_giotto_object,values='normalized',output='matrix')
  ExprSubset<-norm_exp[Sig_scran,]

  Sig_exp<-NULL
  for (i in unique(id)){
    print(i)
    selected_cell_id = cell_metadata$cell_ID[which(cell_metadata$curated_cell_type==i)]
    print(selected_cell_id)
    Sig_exp<-cbind(Sig_exp,(apply(ExprSubset,1,function(y) mean(y[selected_cell_id]))))
  }
  colnames(Sig_exp)<-unique(id)

  if(save==TRUE){
      write.table(as.data.frame(as.matrix(ExprSubset)),file = paste0(out_dir,"/",project,"_expr.txt"),row.names = TRUE,sep="\t",col.names=TRUE,quote =FALSE)
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