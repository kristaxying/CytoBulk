import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
from scipy.sparse import csr_matrix
from utils import *
from cell2location.utils.filtering import filter_genes
from cell2location.models import RegressionModel
import cell2location
from docopt import docopt
'''
def run_cell2location(sc_adata,st_adata):
    
    cell2location.models.RegressionModel.setup_anndata(adata=sc_adata,
                        # cell type, covariate used for constructing signatures
                        labels_key='celltype_major',
                       )
'''
def main(arguments):
    sc_adata = arguments.get("--sc_data")
    st_adata = arguments.get("--st_data")
    #st_adata = sc.datasets.visium_sge(sample_id=r"D:\project\pathway\spatalk_plus\analysis\data\ER+\CID4535")
    #run_cell2location(sc_adata,st_adata)
    

    







if __name__=="__main__":
    arguments = docopt(__doc__, version="test 1.0.0")
    main(arguments)

