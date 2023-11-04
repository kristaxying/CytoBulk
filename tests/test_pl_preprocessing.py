import pytest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import anndata as ad
import pandas as pd
import cytobulk as ct
import scanpy as sc
from string import ascii_uppercase

# tested
@pytest.mark.skip
def test_read_adata(adata_path):
    return sc.read_h5ad(adata_path)

@pytest.mark.skip
def test_read_df(data_path):
    return pd.read_csv(data_path,index_col=0,sep='\t')

# tested
@pytest.mark.skip
@pytest.mark.parametrize("adata_path", [("../data/A36_sample.h5ad")])
def test_sc_qc(adata_path):
    sc = test_read_adata(adata_path)
    assert isinstance(ct.pp.qc_sc(sc,save=True,out_dir='../data',project='A36_sample'),ad.AnnData)
    #return ct.pp.qc_sc(sc)   
@pytest.mark.skip
@pytest.mark.parametrize("adata_path,bulk_path", [("../data/A36_sample.h5ad","../data/reference_bulk_data/A35_sample_stimulated_bulk.txt")])
def test_qc_bulk_sc(bulk_path,adata_path):
    sc = test_read_adata(adata_path)
    bulk = test_read_df(bulk_path)
    ct.pp.qc_bulk_sc(bulk_data = bulk,sc_adata = sc,save=True,out_dir='../data',dataset_name='A36_sc_35_bulk')



@pytest.mark.parametrize("adata_path,bulk_path,marker_path,annotation_key", [("../data/filtered_A36_sample.h5ad",
                                                                            "../data/reference_bulk_data/A35_sample_stimulated_bulk.txt",
                                                                            "../data/cell_meta.xlsx",
                                                                            "Manually_curated_celltype")])
def test_preprocessing_bulk_sc(bulk_path,adata_path,marker_path,annotation_key):
    sc = test_read_adata(adata_path)
    bulk = test_read_df(bulk_path)
    marker = pd.read_excel(marker_path, sheet_name = "marker")
    names = pd.read_excel(marker_path, sheet_name = "rename")
    names = names.set_index(['Original_name'])['Curated_name'].to_dict()
    ct.pp.preprocessing(bulk_data = bulk,sc_adata = sc,marker_data=marker,
                        annotation_key =annotation_key,
                        rename = names,
                        out_dir='../data',dataset_name='filtered_A36_sc_35')
    


if __name__ == '__main__':
    pytest.main(["-s", "test_pl_preprocessing.py"])