import pytest
import os
import sys
import numpy as np
import anndata as ad
import pandas as pd
import cytobulk as ct
import scanpy as sc

@pytest.mark.skip
def test_read_adata(adata_path):
    return sc.read_h5ad(adata_path)

@pytest.mark.parametrize("sc_adata,image_dir,out_dir,project", [(
        "C:/Users/wangxueying/project/CytoBulk/case/he_image/svs/TCGA_LUSC/sub_HTAN_MSK.h5ad",
        "C:/Users/wangxueying/project/CytoBulk/case/he_image/svs/input/TCGA-37-4132_sub2_split",
        "C:/Users/wangxueying/project/CytoBulk/case/he_image/svs/out/TCGA-37-4132_sub2",
        "TCGA-37-4132")])
def test_he_prediction(sc_adata,image_dir,out_dir,project):
    sc_adata = test_read_adata(sc_adata)
    mapping = {
    'plasma cells': 'Plasma Cells',
    'connective tissue': 'Connective Tissue',
    'epithelial': 'Epithelial Cells',
    'neutrophils': 'Neutrophils',
    'lymphocytes': 'Lymphocytes'}

    sc_adata.obs['he_cell_type'] = sc_adata.obs['he_cell_type'].replace(mapping)

    ct.tl.he_mapping(sc_adata = sc_adata,
                     image_dir = image_dir,
                    out_dir =out_dir,
                    project= project,
                    lr_data="C:/Users/wangxueying/project/CytoBulk/case/he_image/svs/input/lrpairs.csv",
                    annotation_key="he_cell_type")
