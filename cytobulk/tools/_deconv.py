
import pandas as pd
from . import model
from .. import get
from .. import utils
from .. import preprocessing as pp
from ._mapping import bulk_mapping
from os.path import exists
import json
import time
import scanpy as sc


def _bulk_sc_deconv(bulk_adata, 
                    presudo_bulk, 
                    sc_adata,
                    marker_dict,
                    annotation_key = "cell_type", 
                    dataset_name = "",
                    counts_location="batch_effected", 
                    out_dir="."):
    """
    Deconvolute the cell type fraction from bulk expression data with single cell dataset as reference.

    Parameters
    ----------
    bulk_adata: anndata.AnnData
        
    **kwargs: parameters in _filter_adata function.
        
    Returns
    -------
    Returns the preprocessed bulk data (adata) , stimualted bulk data and sc data (adata).
    """
    start_t = time.perf_counter()
    deconv = model.GraphDeconv(mode="training")
    training_data = get.count_data_t(presudo_bulk,counts_location=counts_location)
    training_fraction = get.meta(presudo_bulk,position_key="obs")
    test_data = get.count_data_t(bulk_adata,counts_location=counts_location)
    print("=================================================================================================")
    print('Start to train model.')
    deconv.train(out_dir=out_dir+'/model',
                expression=training_data,
                fraction=training_fraction,
                marker=marker_dict,
                sc_adata = sc_adata,
                annotation_key = annotation_key)
    print(f'Time to finish training model: {round(time.perf_counter() - start_t, 2)} seconds')
    print("=================================================================================================")
    print("=================================================================================================")
    print('Start to predict')
    start_t = time.perf_counter()
    deconv_result = deconv.fit(
                out_dir=out_dir+'/output',
                expression=test_data,
                marker=marker_dict,
                sc_adata = sc_adata,
                annotation_key = annotation_key,
                model_folder=out_dir+'/model',
                project = dataset_name)
    print(f'Time to finish deconvolution: {round(time.perf_counter() - start_t, 2)} seconds')
    print("=================================================================================================")
    return deconv_result

def bulk_deconv(bulk_data,
                sc_adata,
                annotation_key,
                marker_data=None,
                rename=None,
                dataset_name="",
                out_dir='.',
                different_source=True,
                cell_list=None,
                scale_factors=10000,
                trans_method="log",
                save = True,
                save_figure=True,
                mapping_sc=True,
                n_cell=100,
                **kwargs):

    if exists(f'{out_dir}/filtered/pseudo_bulk_{dataset_name}.h5ad') and exists(f'{out_dir}/filtered/sc_data_{dataset_name}.h5ad') and \
    exists(f'{out_dir}/filtered/bulk_data_{dataset_name}.h5ad') and exists(f'{out_dir}/filtered/marker_dict.json'):
        
        pseudo_bulk = sc.read_h5ad(f"{out_dir}/filtered/pseudo_bulk_{dataset_name}.h5ad")
        sc_adata = sc.read_h5ad(f"{out_dir}/filtered/sc_data_{dataset_name}.h5ad")
        bulk_adata = sc.read_h5ad(f"{out_dir}/filtered/bulk_data_{dataset_name}.h5ad")
        with open(f"{out_dir}/filtered/marker_dict.json") as json_file:
            marker_dict = json.load(json_file)
    else:
        sc_adata, pseudo_bulk, bulk_adata, marker_dict = pp.preprocessing(bulk_data,
                                                                            sc_adata,
                                                                            annotation_key,
                                                                            marker_data=marker_data,
                                                                            rename=rename,
                                                                            dataset_name=dataset_name,
                                                                            out_dir=out_dir,
                                                                            different_source=different_source,
                                                                            cell_list=cell_list,
                                                                            scale_factors=scale_factors,
                                                                            trans_method=trans_method,
                                                                            save = save,
                                                                            save_figure=save_figure,
                                                                            **kwargs)
    deconv_result = _bulk_sc_deconv(bulk_adata, 
                                    pseudo_bulk, 
                                    sc_adata, 
                                    marker_dict,
                                    annotation_key = annotation_key, 
                                    dataset_name=dataset_name, 
                                    out_dir=out_dir)
    if mapping_sc:
        bulk_adata,sc_mapping_dict = bulk_mapping(deconv_result,
                                                sc_adata,
                                                bulk_adata,
                                                n_cell,
                                                annotation_key)
        bulk_adata.obsm['mapping_dict'] = pd.DataFrame(sc_mapping_dict)

    return deconv_result,bulk_adata

