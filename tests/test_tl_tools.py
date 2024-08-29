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


@pytest.mark.skip
def test_read_csv(data_path):
    return pd.read_csv(data_path,index_col=0,sep=',')


@pytest.mark.skip
@pytest.mark.parametrize("adata_path,bulk_path,marker_path,annotation_key", [("../data/filtered_A36_sample.h5ad",
                                                                            "../data/reference_bulk_data/A35_sample_stimulated_bulk.txt",
                                                                            "../data/cell_meta.xlsx",
                                                                            "curated_cell_type")])
def test_bulk_deconv(bulk_path,adata_path,marker_path,annotation_key):
    sc = test_read_adata(adata_path)
    bulk = test_read_df(bulk_path)
    marker = pd.read_excel(marker_path, sheet_name = "marker")
    names = pd.read_excel(marker_path, sheet_name = "rename")
    names = names.set_index(['Original_name'])['Curated_name'].to_dict()
    ct.tl.bulk_deconv(bulk_data = bulk,sc_adata = sc,marker_data=marker,
                        annotation_key =annotation_key,
                        rename = names,
                        out_dir='D:/project/CytoBulk/case/st_human_sc',dataset_name='filtered_A36_sc_35')


@pytest.mark.skip
@pytest.mark.parametrize("frac_data,sc_adata,bulk_adata,n_cell,annotation_key", [("../data/output/prediction_frac.csv",
                                                                            "../data/filtered/sc_data_filtered_A36_sc_35.h5ad",
                                                                            "../data/filtered/bulk_data_filtered_A36_sc_35.h5ad",
                                                                            100,
                                                                            "curated_cell_type")])
def test_bulk_mapping(frac_data,sc_adata,bulk_adata,n_cell,annotation_key):
    frac_data = test_read_csv(frac_data)
    sc_adata = test_read_adata(sc_adata)
    bulk_adata = test_read_adata(bulk_adata)
    ct.tl.bulk_mapping(frac_data = frac_data,
                        sc_adata = sc_adata,
                        bulk_adata = bulk_adata,
                        n_cell = n_cell,
                        annotation_key=annotation_key,
                        dataset_name="filtered_A36_sc_35",
                        out_dir="../data")


@pytest.mark.skip
@pytest.mark.parametrize("adata_path,bulk_path,marker_path,annotation_key,out_dir,dataset_name", [("../data/input_data/filtered_A36_sample_sc.h5ad",
                                                                            "../data/input_data/filtered_A35_sample_st.h5ad",
                                                                            "../data/input_data/cell_meta.xlsx",
                                                                            "Manually_curated_celltype",
                                                                            './human_sc_st',
                                                                            'human_st')])
def test_st_deconv(bulk_path,adata_path,marker_path,annotation_key,out_dir,dataset_name):
    sc = test_read_adata(adata_path)
    st = test_read_adata(bulk_path)
    marker = pd.read_excel(marker_path, sheet_name = "marker")
    names = pd.read_excel(marker_path, sheet_name = "rename")
    names = names.set_index(['Original_name'])['Curated_name'].to_dict()
    ct.tl.st_deconv(st_adata = st,sc_adata = sc,marker_data=marker,
                        annotation_key =annotation_key,
                        rename = names,
                        out_dir=out_dir,
                        dataset_name=dataset_name)
    


@pytest.mark.skip
@pytest.mark.parametrize("adata_path,bulk_path,marker_path,annotation_key,out_dir,dataset_name", [("D:/project/CytoBulk/CytoBulk/tests/ST_HNSC_GSE139324/input_data/HNSC_GSE139324.h5ad",
                                                                            "D:/project/CytoBulk/CytoBulk/tests/ST_HNSC_GSE139324/input_data/HNSC_GSE139324_expression_test.csv",
                                                                            "D:/project/CytoBulk/CytoBulk/tests/ST_HNSC_GSE139324/input_data/cell_meta.xlsx",
                                                                            "Celltype..minor.lineage.",
                                                                            './ST_HNSC_GSE139324',
                                                                            'HNSC_GSE139324')])
def test_st_deconv_hnsc(bulk_path,adata_path,marker_path,annotation_key,out_dir,dataset_name):
    sc_adata = test_read_adata(adata_path)
    sc_adata.__dict__['_raw'].__dict__['_var'] = sc_adata.__dict__['_raw'].__dict__['_var'].rename(columns={'_index': 'features'})
    st = pd.read_csv(bulk_path,index_col=0)
    st_adata = sc.AnnData(st)
    marker = pd.read_excel(marker_path, sheet_name = "marker")
    names = pd.read_excel(marker_path, sheet_name = "rename")
    names = names.set_index(['Original_name'])['Curated_name'].to_dict()
    ct.tl.st_deconv(st_adata = st_adata,sc_adata = sc_adata,marker_data=marker,
                        annotation_key =annotation_key,
                        rename = names,
                        out_dir=out_dir,
                        dataset_name=dataset_name)

@pytest.mark.skip
@pytest.mark.parametrize("adata_path,bulk_path,marker_path,annotation_key,out_dir,dataset_name", [("../data/input_data/filtered_A36_sample_sc.h5ad",
                                                                            "../data/input_data/stimulated_MM_st_10.h5ad",
                                                                            "../data/input_data/cell_meta.xlsx",
                                                                            "Manually_curated_celltype",
                                                                            './human_sc_st',
                                                                            'human_st')])
def test_st_deconv(bulk_path,adata_path,marker_path,annotation_key,out_dir,dataset_name):
    sc = test_read_adata(adata_path)
    st = test_read_adata(bulk_path)
    marker = pd.read_excel(marker_path, sheet_name = "marker")
    names = pd.read_excel(marker_path, sheet_name = "rename")
    names = names.set_index(['Original_name'])['Curated_name'].to_dict()
    ct.tl.st_deconv(st_adata = st,sc_adata = sc,marker_data=marker,
                        annotation_key =annotation_key,
                        rename = names,
                        out_dir=out_dir,
                        dataset_name=dataset_name)


@pytest.mark.skip
@pytest.mark.parametrize("sc_adata,marker_path,annotation_key,out_dir,dataset_name", [("../data/input_data/MM_GSE151310.h5ad",
                                                                            "../data/input_data/cell_meta.xlsx",
                                                                            "Celltype..minor.lineage.",
                                                                            './MM_25',
                                                                            'MM')])
def test_simulation_st(sc_adata,dataset_name, out_dir, annotation_key,marker_path):
    sc = test_read_adata(sc_adata)
    marker = pd.read_excel(marker_path, sheet_name = "marker")
    names = pd.read_excel(marker_path, sheet_name = "rename")
    names = names.set_index(['Original_name'])['Curated_name'].to_dict()
    sc.__dict__['_raw'].__dict__['_var'] = sc.__dict__['_raw'].__dict__['_var'].rename(columns={'_index': 'features'})
    common_cell = names.keys()
    pseudo_st = ct.ul.st_simulation_case(
                sc, 
                common_cell, 
                annotation_key = annotation_key,
                project=dataset_name, 
                out_dir=out_dir,
                n_sample_each_group=100,
                min_cells_each_group=25,
                cell_gap_each_group=1,
                group_number=5,
                rename_dict=names,
                return_adata=True,
                save=True)

@pytest.mark.skip
@pytest.mark.parametrize("sc_adata,st_adata,marker_path,annotation_key,out_dir,dataset_name", [("../data/input_data/MM_GSE151310.h5ad",
                                                                            "../data/input_data/stimulated_MM_st_5.h5ad",
                                                                            "../data/input_data/cell_meta.xlsx",
                                                                            "Celltype..minor.lineage.",
                                                                            './MM_5_bulk',
                                                                            'MM_5_bulk')])   
def test_st_mapping(sc_adata,st_adata,marker_path,annotation_key,out_dir,dataset_name):
    sc = test_read_adata(sc_adata)
    st = test_read_adata(st_adata)
    sc.__dict__['_raw'].__dict__['_var'] = sc.__dict__['_raw'].__dict__['_var'].rename(columns={'_index': 'features'})
    marker = pd.read_excel(marker_path, sheet_name = "marker")
    names = pd.read_excel(marker_path, sheet_name = "rename")
    names = names.set_index(['Original_name'])['Curated_name'].to_dict()
    ct.tl.st_deconv(st_adata = st,sc_adata = sc,marker_data=marker,
                        annotation_key =annotation_key,
                        rename = names,
                        out_dir=out_dir,
                        dataset_name=dataset_name,
                        different_source=True,
                        n_cell=5)


@pytest.mark.parametrize("st_path", [("E:/CytoBulk/plot_data/BRCA_case/10x/case_data/st_adata_sub_6.h5ad")])   
def test_segmentation(st_path):
    ret,cell_pos = ct.tl.predict_cell_num(
                    st_path,
                    diameter=0,
                    save_png_result=False,
                    model_type='cyto3',
                    cellprob_threshold=.8
                )





if __name__ == '__main__':
    pytest.main(["-s", "test_tl_tools.py"])