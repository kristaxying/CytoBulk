import pandas as pd
import numpy as np
from graph_deconv import *
from image_prediction import *
from utils import *




class CytoBulk:
    def __init__(self,cell_num=100,nor_strategy="tpm"):
        """
            Init the CytoBulk object, the default cell number will be mapped for each sample is 100.
        args:
            cell_num: int, the number of cell for each bulk sample.
        """
        self.cell_num = cell_num
        self.nor_strategy = nor_strategy

    
    def bulk_deconv(self,bulk_exp,ref_sc):
        """
            Input the expression for each data sample, then get the cell type fraction and mapped bulk expression using ref_sc.
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
            Input the HE image for the sample.
        args:
            he_image: HE image.
            selected_region: The region will be mapped with single cell.
        return:
            
        """
        self.image = image_preprocessing(he_image,selected_region)
        image = HEPrediction()
        self.image_cell_type = image.get_cell_type(self.image)
        self.expression = image.get_expression(self.image)
        
    def spatial_mapping(self,st_data=NULL):
        """
            Input the HE image for the sample.
        args:
            st_data: dataframe, the spot expression data.
        return:
            
        """
        self.st = st_preprocessing(st_data)
