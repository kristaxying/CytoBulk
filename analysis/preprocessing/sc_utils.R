library(MAESTRO)
library(Seurat)
library(ggplot2)



Level1_RNAAnnotateCelltype<-function(seurat_object, dir_name, min.score = 0){

  setwd(dir_name)
  
  if("assign.curated" %in% colnames(seurat_object$RNA@meta.data)){
    cluster_id=as.integer(unique(seurat_object$RNA@meta.data$seurat_clusters))-1
    level1_annotation=data.frame(cbind(cluster_id,seurat_object$RNA@meta.data$assign.CIBERSORT[match(cluster_id,seurat_object$RNA@meta.data$seurat_clusters)]))
    names(level1_annotation)=c("cluster_id","level2_anno")
    
    level2_immu=c("B","CD4Tconv","CD8T","CD8Tex","DC","Mast","Mono/Macro","TMKI67","Neutrophils","NK","Plasma","Treg","pDC","EryPro","GMP","Progenitor","Promonocyte","HSC","ILC","hematopoietic stem/progenitor cell")
    malignant_cell=c("Malignant","OPC-like Malignant","AC-like Malignant","OC-like Malignant","NB-like Malignant","Pvalb-like Maligant","Vip-like Malignant","Astro-like Malignant","Endo-like Malignant")
    others_cell=c("Oligodendrocyte","Oligodendrocytes","Neuron","OPC","Alveolar", "Endocrine","Others","Stellate","Astocyte","Acinar","Ductal","Astrocytes","Acinar cells","Microglia","Oligo Opalin")
    
    level1_annotation=as.matrix(level1_annotation)
    level1_annotation[which(!(level1_annotation[,2] %in% c(level2_immu,malignant_cell,others_cell))),2]="Stromal cells"
    level1_annotation[which(level1_annotation[,2] %in% level2_immu),2]="Immune cells"
    level1_annotation[which(level1_annotation[,2] %in% malignant_cell ),2]="Malignant cells"
    level1_annotation[which(level1_annotation[,2] %in% others_cell),2]="Others"
    level2_ann=seurat_object$RNA@meta.data$assign.curated[match(cluster_id,seurat_object$RNA@meta.data$seurat_clusters)]
    print(cbind(level1_annotation,level2_ann))
    
    current.cluster.ids = level1_annotation[,1]
    new.cluster.ids = level1_annotation[,2]
    seurat_object$RNA@meta.data$assign.level1_anno = Idents(seurat_object$RNA)[rownames(seurat_object$RNA@meta.data)]
    seurat_object$RNA@meta.data$assign.level1_anno = plyr::mapvalues(x = seurat_object$RNA@meta.data$assign.level1_anno,
                                                                     from = current.cluster.ids, to = new.cluster.ids)
    
  }
  return(seurat_object)
}

Level3_Tcell_RNAAnnotateCelltype <- function(subset_Tcell,genes,seurat_object){
  
  signatures = read.table("Signature/Tcell_subset_signature.txt",header = F,sep="\t")
  
  celltypes <- as.character(unique(signatures[,1]))
  signature_list <- sapply(1:length(celltypes),function(x){
    return(toupper(as.character(signatures[which(signatures[,1]==celltypes[x]),2])))
  })
  names(signature_list) <- celltypes
  cluster_celltypes = sapply(as.numeric(as.vector(unlist(subset_Tcell$cluster_id))), function(x){
    if(subset_Tcell[match(x,as.numeric(as.vector(unlist(subset_Tcell$cluster_id)))),2] %in% c("CD4Tconv")){
      temp_signature_list=signature_list[c(2,5,7,10:12)]
    }else{
      temp_signature_list=signature_list[c(1,3,4,6,8,13)]
    }
    print(temp_signature_list)
    idx = genes$cluster==x
    avglogFC = genes$avg_logFC[idx]
    names(avglogFC) = toupper(genes$gene[idx])
    score_cluster = sapply(temp_signature_list, function(y){
      score = sum(avglogFC[y], na.rm = TRUE) / log2(length(y))
      return(score)
    })
    print(score_cluster)
    
    if(max(score_cluster, na.rm = TRUE)>0){
      cluster_celltypes = names(score_cluster)[which.max(score_cluster)]
    }else{
      index=which(seurat_object$RNA@meta.data$seurat_clusters == x)[1]
      cluster_celltypes = seurat_object$RNA@meta.data$assign.curated[index]}
  })
  
  return(cbind(as.numeric(as.vector(unlist(subset_Tcell$cluster_id))),cluster_celltypes))
}


Level3_DC_RNAAnnotateCelltype <- function(subset_DC,genes,seurat_object){
  signatures = read.table("Signature/DC_subset_signature.txt",header = F,sep="\t")
  
  celltypes <- as.character(unique(signatures[,1]))
  signature_list <- sapply(1:length(celltypes),function(x){
    return(toupper(as.character(signatures[which(signatures[,1]==celltypes[x]),2])))
  })
  names(signature_list) <- celltypes
  print(as.integer(unique(subset_DC$seurat_clusters))-1)
  signature_list=signature_list[1:5]#exclude pDC 
  cluster_celltypes = sapply(as.integer(unique(subset_DC$seurat_clusters)), function(x){
    idx = genes$cluster==x
    avglogFC = genes$avg_logFC[idx]
    names(avglogFC) = toupper(genes$gene[idx])
    score_cluster = sapply(signature_list, function(y){
      score = sum(avglogFC[y], na.rm = TRUE) / log2(length(y))
      print(score)
      return(score)
    })
    if(max(score_cluster, na.rm = TRUE)>0){
      cluster_celltypes = names(score_cluster)[which.max(score_cluster)]
    }else{
      index=which(seurat_object$RNA@meta.data$seurat_clusters == x)[1]
      cluster_celltypes = seurat_object$RNA@meta.data$assign.curated[index]}
  })
  
  return(cbind(as.integer(unique(subset_DC$seurat_clusters)),cluster_celltypes))
}


Level3_RNAAnnotateCelltype<-function(seurat_object,dir_name){
  setwd(dir_name)
  genes<-seurat_object$genes
  if(length(seurat_object$RNA@meta.data$assign.curated) > 0){
    cluster_id=as.integer(unique(seurat_object$RNA@meta.data$seurat_clusters))
    level3_annotation=data.frame(cbind(cluster_id,seurat_object$RNA@meta.data$assign.curated[match(cluster_id,seurat_object$RNA@meta.data$seurat_clusters)]))
    names(level3_annotation)=c("cluster_id","level2_anno")
    print(level3_annotation)
    #subset_Tcell=subset(seurat_object$RNA@meta.data,assign.curated %in% c("CD8T","CD4Tconv"))
    subset_Tcell=subset(level3_annotation,level2_anno %in% c("CD8T","CD4Tconv"))
    Tcell_annotation=Level3_Tcell_RNAAnnotateCelltype(subset_Tcell,genes,seurat_object)
    print(Tcell_annotation)
    subset_DC=subset(seurat_object$RNA@meta.data,assign.curated %in% c("Mono/Macro","DC"))
    DC_Mo_annotation=Level3_DC_RNAAnnotateCelltype(subset_DC,genes,seurat_object)
    print(DC_Mo_annotation)
    level3_anno=rbind(Tcell_annotation,DC_Mo_annotation)
    level3_annotation=as.matrix(level3_annotation)
    if(dim(level3_anno)[1] > 1){
      level3_anno=matrix(unlist(level3_anno),byrow=F,ncol=2)
      level3_annotation[match(level3_anno[,1],level3_annotation[,1]),2]=level3_anno[,2]
    }else{
      level3_annotation[match(unlist(level3_anno[1]),level3_annotation[,1]),2]=unlist(level3_anno[2])
    }
    current.cluster.ids = level3_annotation[,1]
    new.cluster.ids = level3_annotation[,2]
    seurat_object$RNA@meta.data$assign.level3_anno = Idents(seurat_object$RNA)[rownames(seurat_object$RNA@meta.data)]
    seurat_object$RNA@meta.data$assign.level3_anno = plyr::mapvalues(x = seurat_object$RNA@meta.data$assign.level3_anno,
                                                                     from = current.cluster.ids, to = new.cluster.ids)

    return(seurat_object)
  }
}

