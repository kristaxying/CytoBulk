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


@pytest.mark.skip
def test_read_adata(adata_path):
    return sc.read_h5ad(adata_path)


@pytest.mark.parametrize("adata_path, project, out_dir,annotation_key,marker_path", [("../data/A35_sample.h5ad","A35_sample","../data","Manually_curated_celltype",
                                                                                    "../data/cell_meta.xlsx")])

@pytest.mark.skip
def test_stimulation_bulk(adata_path,project, out_dir, annotation_key,marker_path):
    sc = test_read_adata(adata_path)
    marker = pd.read_excel(marker_path, sheet_name = "marker")
    names = pd.read_excel(marker_path, sheet_name = "rename")
    names = names.set_index(['Original_name'])['Curated_name'].to_dict()
    common_cell = names.keys()
    pseudo_bulk, pseudo_prop = ct.ul.bulk_simulation(sc, 
                                common_cell, 
                                annotation_key = annotation_key,
                                project=project, 
                                out_dir=out_dir,
                                n_sample_each_group=100,
                                min_cells_each_group=100,
                                cell_gap_each_group=100,
                                group_number=5,
                                rename_dict=names,
                                save=True)
    assert isinstance(pseudo_bulk,pd.DataFrame)
    assert isinstance(pseudo_prop,pd.DataFrame)
    assert len(common_cell)==len(pseudo_prop.columns)

@pytest.mark.parametrize("adata_path, project, out_dir,annotation_key,marker_path", [("../data/A35_sample.h5ad","A35_sample","../data","Manually_curated_celltype",
                                                                                    "../data/cell_meta.xlsx")])
def test_stimulation_case(adata_path,project, out_dir, annotation_key,marker_path):
    sc = test_read_adata(adata_path)
    marker = pd.read_excel(marker_path, sheet_name = "marker")
    names = pd.read_excel(marker_path, sheet_name = "rename")
    names = names.set_index(['Original_name'])['Curated_name'].to_dict()
    common_cell = names.keys()
    pseudo_bulk, pseudo_prop = ct.ul.bulk_simulation_case(
                                sc, 
                                common_cell, 
                                annotation_key = annotation_key,
                                project=project, 
                                out_dir=out_dir,
                                n_sample_each_group=100,
                                min_cells_each_group=100,
                                cell_gap_each_group=100,
                                group_number=5,
                                rename_dict=names,
                                save=True)
    assert isinstance(pseudo_bulk,pd.DataFrame)
    assert isinstance(pseudo_prop,pd.DataFrame)
    assert len(common_cell)==len(pseudo_prop.columns)



if __name__ == '__main__':
    pytest.main(["-s", "test_ul_stimulation.py"])