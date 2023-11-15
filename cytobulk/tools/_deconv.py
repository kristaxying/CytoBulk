
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
        An :class:`~anndata.AnnData` containing the input bulk expression.
    presudo_bulk: anndata.AnnData
        An :class:`~anndata.AnnData` containing the stimulated bulk expression with single cell dataset.
    sc_adata: anndata.AnnData
        An :class:`~anndata.AnnData` containing the single cell expression.
    annotation_key: string
        The `.obs` key where the single cell annotation is stored.: anndata.AnnData.
    marker_data: 
        An :class:`~pandas.dataframe` which columns are cell types, rows are marker gene.
    dataset_name: string.
        The prefix of output file.
    counts_location: string.
        The layer name of the expression in presudo_bulk to train the model.
    out_dir: string, optional
        The path to store the output data.
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
    """
    Deconvolute the cell type fraction from bulk expression data with single cell dataset as reference.
    Reconstruct the bulk data using single cell.

    Parameters
    ----------
    bulk_data: dataframe
        An :class:`~pandas.dataframe` containing the bulk expression data. 
        The first column should be gene symbol, following column should be sample name.
    sc_adata: anndata.AnnData
        An :class:`~anndata.AnnData` containing the single cell expression.
    annotation_key: string
        The `.obs` key where the single cell annotation is stored.: anndata.AnnData.
    marker_data: 
        An :class:`~pandas.dataframe` which columns are cell types, rows are marker gene.
    dataset_name: string.
        The prefix of output file.
    out_dir: string, optional
        The path to store the output data.
    different_source: boolean, optional
        True for single cell and bulk data from the same sample, which means not executing batch effect.
        False for single cell and bulk data from the different samples, which means executing batch effect.
    cell_list: list, optional
        The list indicate the cell type names which need to take into consideration.
    scale_factors: int, optional
        The number of counts to normalize every observation to before computing profiles. If `None`, no normalization is performed. 
    trans_method: string, optional
        What transformation to apply to the expression before computing the profiles. 
        - "log": log(x+1)
        - `None`: no transformation
    save: boolean, optional
        Whether save the result data during each step. If saving, the processing may be skipped.
    save_figure: boolean, optional
        Whether save figures during preprocessing. eg. scatter plot of pca data.
    mapping_sc: boolean, optional
        Whether reconstruct the bulk data with single cell data.
    n_cell: int, optional
        The number of cells within each bulk.
    **kwargs: 
        Additional keyword arguments forwarded to
        :func:`~cytobulk.preprocessing.qc_bulk_sc`.

        
    Returns
    -------
    Returns the deconvolution result and reconstructed bulk.
    """
    # check the filtered dataset. If exist, skipping preprocessing.
    if exists(f'{out_dir}/filtered/pseudo_bulk_{dataset_name}.h5ad') and exists(f'{out_dir}/filtered/sc_data_{dataset_name}.h5ad') and \
    exists(f'{out_dir}/filtered/bulk_data_{dataset_name}.h5ad') and exists(f'{out_dir}/filtered/marker_dict.json'):
        
        pseudo_bulk = sc.read_h5ad(f"{out_dir}/filtered/pseudo_bulk_{dataset_name}.h5ad")
        sc_adata = sc.read_h5ad(f"{out_dir}/filtered/sc_data_{dataset_name}.h5ad")
        bulk_adata = sc.read_h5ad(f"{out_dir}/filtered/bulk_data_{dataset_name}.h5ad")
        with open(f"{out_dir}/filtered/marker_dict.json") as json_file:
            marker_dict = json.load(json_file)
    else:
        #preprocessing
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
    #deconvolution
    deconv_result = _bulk_sc_deconv(bulk_adata, 
                                    pseudo_bulk, 
                                    sc_adata, 
                                    marker_dict,
                                    annotation_key = annotation_key, 
                                    dataset_name=dataset_name, 
                                    out_dir=out_dir)
    if mapping_sc:
        #bulk reconstruction
        bulk_adata,sc_mapping_dict = bulk_mapping(deconv_result,
                                                sc_adata,
                                                bulk_adata,
                                                n_cell,
                                                annotation_key)
        bulk_adata.obsm['mapping_dict'] = pd.DataFrame(sc_mapping_dict)

    return deconv_result,bulk_adata

