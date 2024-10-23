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
@pytest.mark.parametrize("adata_path", [("C:/Users/wangxueying/project/CytoBulk/case/mouse_mob/output/output/mouse_mob_st_adata.h5ad")])
def test_st_pie_plot(adata_path):
    adata = test_read_adata(adata_path)
    ct.plots.celltype_fraction_pie(adata=adata,
                                   scale_facter_x=5,
                                   scale_factor_y=5,
                                   out_dir="C:/Users/wangxueying/project/CytoBulk/case/mouse_mob/output/output",
                                   rotation_angle=225)

@pytest.mark.skip
@pytest.mark.parametrize("adata_path", [("C:/Users/wangxueying/project/CytoBulk/case/mouse_mob/output/output/mouse_mob_st_adata.h5ad")])
def test_st_cell_heatmap_plot(adata_path):
    adata = test_read_adata(adata_path)
    ct.plots.celltype_fraction_heatmap(adata=adata,
                                       label='OSN',
                                        out_dir="C:/Users/wangxueying/project/CytoBulk/case/mouse_mob/output/output",
                                        rotation_angle=225)
    
@pytest.mark.skip
@pytest.mark.parametrize("adata_path", [("C:/Users/wangxueying/project/CytoBulk/case/mouse_mob_1/output/output/mouse_mob_st_adata.h5ad")])
def test_st_cell_heatmap_plot(adata_path):
    adata = test_read_adata(adata_path)
    ct.plots.paired_violin(adata=adata,
                        label='OSN',
                        gene='Kctd12',
                        out_dir="C:/Users/wangxueying/project/CytoBulk/case/mouse_mob_1/output/output")


@pytest.mark.skip  
@pytest.mark.parametrize("adata_path", [("C:/Users/wangxueying/project/CytoBulk/case/MOB_layer_cut_add_7181/output/reconstructed_mouse_mob_st.h5ad")])
def test_st_reconstruction(adata_path):
    adata = test_read_adata(adata_path)
    ct.plots.reconstruction_corr(adata=adata,
                                out_dir="C:/Users/wangxueying/project/CytoBulk/case/MOB_layer_cut_add_7181/output",
                                rotation_angle=225)

@pytest.mark.skip  
@pytest.mark.parametrize("adata_path", [("C:/Users/wangxueying/project/CytoBulk/case/PDAC/out/output/reconstructed_PDAC_st.h5ad")])
def test_st_reconstruction_pdac(adata_path):
    adata = test_read_adata(adata_path)
    ct.plots.reconstruction_corr(adata=adata,
                                out_dir="C:/Users/wangxueying/project/CytoBulk/case/PDAC/out",
                                rotation_angle=225) 
@pytest.mark.skip    
@pytest.mark.parametrize("adata_path", [("C:/Users/wangxueying/project/CytoBulk/case/10x/sub4/output/reconstructed_sub4_st.h5ad")])
def test_st_reconstruction_10x(adata_path):
    adata = test_read_adata(adata_path)
    ct.plots.reconstruction_corr(adata=adata,
                                out_dir="C:/Users/wangxueying/project/CytoBulk/case/10x/sub4",
                                spot_size=1) 
    
@pytest.mark.skip 
@pytest.mark.parametrize("adata_path", [("C:/Users/wangxueying/project/CytoBulk/case/10x/sub6/output/reconstructed_sub6_st.h5ad")])
def test_st_reconstruction_10x_6(adata_path):
    adata = test_read_adata(adata_path)
    ct.plots.reconstruction_corr(adata=adata,
                                out_dir="C:/Users/wangxueying/project/CytoBulk/case/10x/sub6",
                                spot_size=1) 
    

@pytest.mark.parametrize("adata_path", [("C:/Users/wangxueying/project/CytoBulk/case/10x/sub3/output/reconstructed_sub3_st.h5ad")])
def test_st_reconstruction_10x_3(adata_path):
    adata = test_read_adata(adata_path)
    ct.plots.reconstruction_corr(adata=adata,
                                out_dir="C:/Users/wangxueying/project/CytoBulk/case/10x/sub3",
                                spot_size=1) 