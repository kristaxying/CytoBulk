import pytest
import os
import sys
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



@pytest.mark.skip
@pytest.mark.parametrize("sc_adata,st_adata", [("C:/Users/wangxueying/project/CytoBulk/case/mouse_mob/input/sc_layer_mob.h5ad",
                                                "C:/Users/wangxueying/project/CytoBulk/case/MOB_layer_cut_add_7181/output/MOB_layer_st_adata.h5ad")])   
def test_st_mapping_mouse(sc_adata,st_adata):
    sc = test_read_adata(sc_adata)
    st = test_read_adata(st_adata)
    ct.tl.st_mapping(st_adata = st,sc_adata = sc,
                     out_dir="C:/Users/wangxueying/project/CytoBulk/case/MOB_layer_cut_add_7181",
                     project="mouse_mob",
                     annotation_key='subtype',
                     mean_cell_numbers=18)


@pytest.mark.skip
@pytest.mark.parametrize("sc_adata,st_adata", [("C:/Users/wangxueying/project/CytoBulk/case/PDAC/input/sc_adata.h5ad",
                                                "C:/Users/wangxueying/project/CytoBulk/case/PDAC/out/output/PDAC_st_adata.h5ad")])   
def test_st_mapping_pdac(sc_adata,st_adata):
    sc = test_read_adata(sc_adata)
    st = test_read_adata(st_adata)
    ct.tl.st_mapping(st_adata = st,sc_adata = sc,
                     out_dir="C:/Users/wangxueying/project/CytoBulk/case/PDAC/out",
                     project="pdac",
                     annotation_key='cell_type',
                     mean_cell_numbers=18)
    

@pytest.mark.skip
@pytest.mark.parametrize("sc_adata,st_adata", [("C:/Users/wangxueying/project/CytoBulk/case/10x/input/sc_adata.h5ad",
                                                "C:/Users/wangxueying/project/CytoBulk/case/10x/sub4/st_adata_sub_4_construction.h5ad")])   
def test_st_mapping(sc_adata,st_adata):
    sc = test_read_adata(sc_adata)
    st = test_read_adata(st_adata)
    ct.tl.st_mapping(st_adata = st,sc_adata = sc,
                     out_dir="C:/Users/wangxueying/project/CytoBulk/case/10x/sub4",
                     project="sub4",
                     annotation_key='cell_type',
                     mean_cell_numbers=15)
@pytest.mark.skip   
@pytest.mark.parametrize("sc_adata,st_adata", [("C:/Users/wangxueying/project/CytoBulk/case/10x/input/sc_adata.h5ad",
                                                "C:/Users/wangxueying/project/CytoBulk/case/10x/sub6/st_adata_sub_6_deconv.h5ad")])   
def test_st_mapping_sub6(sc_adata,st_adata):
    sc = test_read_adata(sc_adata)
    st = test_read_adata(st_adata)
    ct.tl.st_mapping(st_adata = st,sc_adata = sc,
                     out_dir="C:/Users/wangxueying/project/CytoBulk/case/10x/sub6",
                     project="sub6",
                     annotation_key='cell_type')

@pytest.mark.parametrize("sc_adata,st_adata", [("C:/Users/wangxueying/project/CytoBulk/case/10x/input/sc_adata.h5ad",
                                                r"C:\Users\wangxueying\project\CytoBulk\case\10x\sub4\st_adata_sub_4_construction.h5ad")])   
def test_st_mapping_sub4(sc_adata,st_adata):
    sc = test_read_adata(sc_adata)
    st = test_read_adata(st_adata)
    print(st.uns.keys())
    ct.tl.st_mapping(st_adata = st,sc_adata = sc,
                     out_dir="C:/Users/wangxueying/project/CytoBulk/case/10x/sub4",
                     project="sub4",
                     annotation_key='cell_type')
@pytest.mark.skip      
@pytest.mark.parametrize("sc_adata,st_adata", [("C:/Users/wangxueying/project/CytoBulk/case/10x/input/sc_adata.h5ad",
                                                "C:/Users/wangxueying/project/CytoBulk/case/10x/sub5/st_adata_sub_5_deconv.h5ad")])   
def test_st_mapping_sub5(sc_adata,st_adata):
    sc = test_read_adata(sc_adata)
    st = test_read_adata(st_adata)
    ct.tl.st_mapping(st_adata = st,sc_adata = sc,
                     out_dir="C:/Users/wangxueying/project/CytoBulk/case/10x/sub5",
                     project="sub5",
                     annotation_key='cell_type')
    
@pytest.mark.skip           
@pytest.mark.parametrize("sc_adata,bulk_adata", [("C:/Users/wangxueying/project/CytoBulk/case/human_sc/input/filtered_A36_sample.h5ad",
                                                "C:/Users/wangxueying/project/CytoBulk/case/human_sc/out/output/filtered_A36_sc_35_bulk_adata.h5ad")])   
def test_bulk_mapping_sub(sc_adata,bulk_adata):
    sc = test_read_adata(sc_adata)
    bulk = test_read_adata(bulk_adata)
    ct.tl.bulk_mapping(bulk_adata = bulk,
                       sc_adata = sc,
                        out_dir="C:/Users/wangxueying/project/CytoBulk/case/human_sc/out",
                        project="human_sc_bulk",
                        n_cell=1000,
                        annotation_key='Manually_curated_celltype')
