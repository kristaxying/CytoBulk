import pandas as pd
import numpy as np
from graph_deconv import *
from image_prediction import *
from utils import *
from read_data import *
import os




class CytoBulk:
    def __init__(self,cell_num=100,ref_marker=os.path.join(os.path.split(os.path.realpath(__file__))[0],'config/marker_gene.txt'),out_path='./output'):
        """
            Init the CytoBulk object, the default cell number will be mapped for each sample is 100.
        args:
            cell_num: int, the number of cell for each bulk sample.
            nor_strategy: the strategy to normalize bulk data. default is "log2(tpm)", the other choices include "none" and "log2".
        """
        self.cell_num = cell_num
        self.out_path = out_path

    
    def bulk_deconv(self,
                    gene_len_path='../config/gene_length.txt',
                    mode='prediction',
                    training_sc=None,
                    training_meta=None,
                    sc_nor = True,
                    nor_strategy="log2(tpm)",
                    marker_label = 'self_designed',
                    ref_marker=os.path.join(os.path.split(os.path.realpath(__file__))[0],'config/marker_gene.txt')):
        #bulk_deconv(self, bulk_exp_path, ref_sc_path, gene_len_path='./config/gene_length.txt',mode='prediction',training_sc=None,training_meta=None):
        """
            Input is the expression for each data sample, then get the cell type fraction and mapped bulk expression using ref_sc.
        args:
            bulk_exp_path: the path of bulk rna data.
            ref_sc_path: the path of single cell rederence datasets for mapping.
            gene_len_path: the gene length to make log2tmp normalization.
            mode: the mode to use deconvolution, can select from "prediction" and "training".
            training_sc: the single cell to train the deconv model as reference.
            training_meta: the meta data to classificate cells in training_sc.
            nor_strategy: strategies to normalize bulk data.
            marker_label: the marker gene could be 'self_designed' or 'auto_find'.
            ref_marker: if marker_label = self_designed, THIS 
        return:
            cell_fraction: dataframe, columns is the cell type, rows are sample_name.
            mapped_sc: dataframe, columns is the sample name, rows are GeneSymbol.
        """
        if marker_label == 'self_designed':
            self.ref_marker = read_marker_genes(ref_marker)
        else:
            self.ref_marker = None
        # check output dir
        out_dir = check_paths(self.out_path)
        # if mode = trainging, stimulation and train.
        if mode == "training":
            if training_sc is None:
                raise ValueError("For deconvolution, if using training mode, please provide scRNA-seq data and corresponding cell meta.")
            else:
                print("Please notice that the training mode is chosen")
                sti_bulk,sti_fraction = bulk_simulation(training_sc,training_meta,self.ref_marker,sc_nor,out_dir)
                '''
                please add training part.
                '''
                
        '''
        elif mode == "prediction":
            training_sc,trainging_meta = read_training_data(training_sc,trainging_meta)
        else:
            raise ValueError("For deconvolution, if using training mode, please provide scRNA-seq data and corresponding cell meta.")
        '''

        '''
        self.bulk_exp = read_bulk_data(bulk_exp_path,self.ref_marker)
        # check the bulk rna data format and genesymbol
        check_bulk_format(self.bulk_exp, self.ref_marker)
        if self.nor_strategy=="log2(tpm)":
            self.norm_bulk = bulk_normalization(self.bulk_exp,self.nor_strategy,gene_len_path)
        else:
            self.norm_bulk = self.bulk_exp
        deconv = GraphDeconv()
        self.cell_fraction = deconv.fit(self.normalized_bulk)
        self.mapped_sc = deconv.sc_mapping(self.normalized_bulk,ref_sc_path)
        '''

    def HE_prediction(self,he_image,selected_region = "all"):
        """
            Input is the HE image for the sample.
        args:
            he_image: HE image.
            selected_region: The region will be mapped with single cell.
        return:
            image_cell_type: dataframe, columns are the cell x y, spot x y and cell type, rows are cell id.
            expression: dataframe, the first two rows are spot x, spot y and the followed rows are GeneSymbol.
        """
        self.image = image_preprocessing(he_image,selected_region)
        image = HEPrediction()
        self.image_cell_type = image.get_cell_type(self.image)
        self.image_exp = image.get_expression(self.image)

    def set_image_exp(self,st_data):
        self.image_exp = st_data
    
    def set_st_meta(self,st_meta):
        self.st_meta = st_meta

    def set_sc_meta(self,sc_meta):
        self.sc_meta = sc_meta
    
    def set_sc_data(self,sc_data):
        self.sc_data = sc_data
    

        
    def spatial_mapping(self,st_data=None,st_nor_strategy="scanpy"):
        """
            Input are the HE expression data, HE cell type data and the referenced single cell dataset.
        args:
            st_data: dataframe, the spot expression data.
            st_nor_strategy: normalization strategy of spot data
        return:
            mapped_st: dataframe, the first two rows are cell x, cell y and the followed rows are GeneSymbol.
        """
        if st_data:
            self.st_exp = st_preprocessing(st_data,st_nor_strategy)
            self.st_exp = st_imputation(st_data,self.image_exp)
        else:
            self.st_exp = self.image_exp

        M = SpatialMapping(self.image_exp,self.st_meta,self.sc_data,self.sc_meta)
        spot_cell_type = M.preprocessing()
        #self.sc_st = mapping.get_st_mapping(self.st_exp,self.mapped_sc,self.image_cell_type)
        #self.mapped_st = mapping.bipartite_matching(self.sc_st,self.image_sc)
    
    def plot_mapped_st(self):
        """
            Input are the HE image and self.mapped_st.
        return:
            Dot plot for mapped HE.
        """


