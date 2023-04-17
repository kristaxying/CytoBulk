import pandas as pd
import numpy as np
from graph_deconv import *
from image_prediction import *
from utils import *
from spatial_mapping import *




class CytoBulk:
    def __init__(self,cell_num=100,nor_strategy="tpm_log"):
        """
            Init the CytoBulk object, the default cell number will be mapped for each sample is 100.
        args:
            cell_num: int, the number of cell for each bulk sample.
        """
        self.cell_num = cell_num
        self.nor_strategy = nor_strategy

    
    def bulk_deconv(self,bulk_exp,ref_sc):
        """
            Input is the expression for each data sample, then get the cell type fraction and mapped bulk expression using ref_sc.
        args:
            bulk_exp: dataframe, columns is the sample name, rows are GeneSymbol.
            ref_sc: dataframe, columns is the barcode_cell type rows are GeneSymbol.
        return:
            cell_fraction: dataframe, columns is the cell type, rows are sample_name.
            mapped_sc: dataframe, columns is the sample name, rows are GeneSymbol.
        """
        self.bulk_exp = bulk_exp
        self.normalized_bulk = bulk_normalization(bulk_exp,self.nor_strategy)
        deconv = GraphDeconv()
        self.cell_fraction = deconv.fit(self.normalized_bulk)
        self.mapped_sc = deconv.sc_mapping(self.normalized_bulk,ref_sc)


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

        mapping = SpatialMapping()
        self.sc_st = mapping.get_st_mapping(self.st_exp,self.mapped_sc,self.image_cell_type)
        self.mapped_st = mapping.bipartite_matching(self.sc_st,)
    
    def plot_mapped_st(self):
        """
            Input are the HE image and self.mapped_st.
        return:
            Dot plot for mapped HE.
        """


