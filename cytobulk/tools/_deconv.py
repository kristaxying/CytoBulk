
import pandas as pd
from . import model
from .. import preprocessing as pp
from os.path import exists
import json
import time
import scanpy as sc


def _bulk_sc_deconv(bulk_adata, 
                    presudo_bulk, 
                    sc_adata,
                    common_cells,
                    annotation_key = "cell_type", 
                    dataset_name = "",
                    counts_location="batch_effected", 
                    out_dir=".",
                    batch_effect=True,
                    is_st=False):
    """
    Deconvolute the cell type fraction from bulk expression data with single cell dataset as reference.

    Parameters
    ----------
    bulk_adata : anndata.AnnData
        An :class:`~anndata.AnnData` containing the input bulk expression.
    presudo_bulk : anndata.AnnData
        An :class:`~anndata.AnnData` containing the stimulated bulk expression with single cell dataset.
    sc_adata : anndata.AnnData
        An :class:`~anndata.AnnData` containing the single cell expression.
    annotation_key : string
        The `.obs` key where the single cell annotation is stored. : anndata.AnnData.
    marker_data : 
        An :class:`~pandas.dataframe` which columns are cell types, rows are marker gene.
    dataset_name : string.
        The prefix of output file.
    counts_location : string.
        The layer name of the expression in presudo_bulk to train the model.
    out_dir : string, optional
        The path to store the output data.
    **kwargs : parameters in _filter_adata function.
        
    Returns
    -------
    Returns the preprocessed bulk data (adata) , stimualted bulk data and sc data (adata).

    """
    start_t = time.perf_counter()
    deconv = model.GraphDeconv(mode="training")
    #training_data = get.count_data_t(presudo_bulk,counts_location=counts_location)
    #training_fraction = get.meta(presudo_bulk,position_key="obs")
    print("=================================================================================================")
    print('Start to train model.')
    model_dir = out_dir + '/st_model' if is_st else out_dir + '/model'
    deconv.train(out_dir=model_dir,
                presudo_bulk=presudo_bulk,
                bulk_adata=bulk_adata,
                cell_list=common_cells,
                sc_adata = sc_adata,
                annotation_key = "curated_cell_type",
                project_name = dataset_name,
                data_num=bulk_adata.n_obs*5,
                batch_effect=batch_effect,
                is_st=is_st)
    print(f'Time to finish training model: {round(time.perf_counter() - start_t, 2)} seconds')
    print("=================================================================================================")
    '''
    training_data = get.count_data_t(presudo_bulk,counts_location=counts_location)
    training_fraction = get.meta(presudo_bulk,position_key="obs")
    test_data = get.count_data_t(bulk_adata,counts_location=counts_location)
    print("=================================================================================================")
    print('Start to train model.')
    model_dir = out_dir + '/st_model' if is_st else out_dir + '/model'
    deconv.train(out_dir=model_dir,
                expression=training_data,
                fraction=training_fraction,
                marker=marker_dict,
                sc_adata = sc_adata,
                annotation_key = annotation_key,
                is_st=is_st)
    print(f'Time to finish training model: {round(time.perf_counter() - start_t, 2)} seconds')
    print("=================================================================================================")
    '''
    print("=================================================================================================")
    print('Start to predict')
    #test_data = get.count_data_t(bulk_adata)
    start_t = time.perf_counter()
    deconv_result = deconv.fit(
                out_dir=out_dir+'/output',
                expression=bulk_adata,
                cell_list=common_cells,
                sc_adata = sc_adata,
                annotation_key = annotation_key,
                model_folder=model_dir,
                file_dir=model_dir,
                project = dataset_name,
                is_st=is_st)
    print(f'Time to finish deconvolution: {round(time.perf_counter() - start_t, 2)} seconds')
    print("=================================================================================================")
    return deconv_result

def bulk_deconv(bulk_data,
                sc_adata,
                annotation_key,
                marker_list=None,
                rename=None,
                dataset_name="",
                out_dir='.',
                different_source=True,
                cell_list=None,
                scale_factors=100000,
                trans_method="log",
                save = True,
                save_figure=True,
                n_cell=2000,
                **kwargs):
    """
    Deconvolute the cell type fraction from bulk expression data with single cell dataset as reference.
    Reconstruct the bulk data using single cell.

    Parameters
    ----------
    bulk_data : dataframe
        An :class:`~pandas.dataframe` containing the bulk expression data. 
        The first column should be gene symbol, following column should be sample name.
    sc_adata : anndata.AnnData
        An :class:`~anndata.AnnData` containing the single cell expression.
    annotation_key : string
        The `.obs` key where the single cell annotation is stored. : anndata.AnnData.
    marker_data : 
        An :class:`~pandas.dataframe` which columns are cell types, rows are marker gene.
    dataset_name : string.
        The prefix of output file.
    out_dir : string, optional
        The path to store the output data.
    different_source : boolean, optional
        True for single cell and bulk data from the same sample, which means not executing batch effect.
        False for single cell and bulk data from the different samples, which means executing batch effect.
    cell_list : list, optional
        The list indicate the cell type names which need to take into consideration.
    scale_factors : int, optional
        The number of counts to normalize every observation to before computing profiles. If `None`, no normalization is performed. 
    trans_method : string, optional
        What transformation to apply to the expression before computing the profiles. 
        - "log" : log(x+1)
        - `None` : no transformation
    save : boolean, optional
        Whether save the result data during each step. If saving, the processing may be skipped.
    save_figure : boolean, optional
        Whether save figures during preprocessing. eg. scatter plot of pca data.
    mapping_sc : boolean, optional
        Whether reconstruct the bulk data with single cell data.
    n_cell : int, optional
        The number of cells within each bulk.
    **kwargs : 
        Additional keyword arguments forwarded to
        :func:`~cytobulk.preprocessing.qc_bulk_sc`.

        
    Returns
    -------
    Returns the deconvolution result and reconstructed bulk.
    """
    # check the filtered dataset. If exist, skipping preprocessing.
    bulk_ori_adata = bulk_data.copy()
    if exists(f'{out_dir}/filtered/pseudo_bulk_{dataset_name}.h5ad') and exists(f'{out_dir}/filtered/sc_data_{dataset_name}.h5ad') and \
    exists(f'{out_dir}/filtered/bulk_data_{dataset_name}.h5ad') and exists(f'{out_dir}/filtered/cells_{dataset_name}.json'):
        
        pseudo_bulk = sc.read_h5ad(f"{out_dir}/filtered/pseudo_bulk_{dataset_name}.h5ad")
        sc_adata = sc.read_h5ad(f"{out_dir}/filtered/sc_data_{dataset_name}.h5ad")
        bulk_adata = sc.read_h5ad(f"{out_dir}/filtered/bulk_data_{dataset_name}.h5ad")
        with open(f"{out_dir}/filtered/cells_{dataset_name}.json") as json_file:
            common_cell = json.load(json_file)
        annotation_key="curated_cell_type"
    else:
        #preprocessing
        sc_adata, pseudo_bulk, bulk_adata, common_cell,annotation_key = pp.preprocessing(bulk_data,
                                                                        sc_adata,
                                                                        annotation_key,
                                                                        is_st=False,
                                                                        marker_list=marker_list,
                                                                        rename=rename,
                                                                        dataset_name=dataset_name,
                                                                        out_dir=out_dir,
                                                                        different_source=different_source,
                                                                        cell_list=cell_list,
                                                                        scale_factors=scale_factors,
                                                                        trans_method=trans_method,
                                                                        n_sample_each_group=len(bulk_data.obs_names)*5,
                                                                        min_cells_each_group=n_cell,
                                                                        cell_gap_each_group=50,
                                                                        group_number=10,
                                                                        save = save,
                                                                        save_figure=save_figure,
                                                                        **kwargs)
    #deconvolution
    if exists(f'{out_dir}/output/{dataset_name}_prediction_frac.csv'):
        deconv_result = pd.read_csv(f'{out_dir}/output/{dataset_name}_prediction_frac.csv',index_col=0)
    else:
        deconv_result = _bulk_sc_deconv(bulk_adata, 
                                        pseudo_bulk, 
                                        sc_adata, 
                                        common_cell,
                                        annotation_key = annotation_key, 
                                        dataset_name=dataset_name, 
                                        out_dir=out_dir)
    row_sums = deconv_result.sum(axis=1)
    df_normalized = deconv_result.div(row_sums, axis=0)
    bulk_ori_adata.uns['deconv']=df_normalized
    df_normalized.to_csv(f"{out_dir}/output/{dataset_name}_prediction_frac_normalized.csv")
    bulk_ori_adata.write_h5ad(f'{out_dir}/output/{dataset_name}_bulk_adata.h5ad')

    return deconv_result,bulk_adata


def st_deconv(st_adata,
            sc_adata,
            annotation_key,
            marker_list=None,
            rename=None,
            dataset_name="",
            out_dir='.',
            different_source=True,
            cell_list=None,
            scale_factors=10000,
            trans_method="log",
            save = True,
            save_figure=True,
            n_cell=10,
            max_cells=40,
            **kwargs):
    """
    Deconvolute the cell type fraction from spot expression data with single cell dataset as reference.

    Parameters
    ----------
    st_adata : anndata.AnnData
        An :class:`~anndata.AnnData` containing the single cell expression.
        The first column should be gene symbol, following column should be spot name.
    sc_adata : anndata.AnnData
        An :class:`~anndata.AnnData` containing the single cell expression.
    annotation_key : string
        The `.obs` key where the single cell annotation is stored. : anndata.AnnData.
    marker_list : 
        An :class:`~pandas.dataframe` which columns are cell types, rows are marker gene.
    dataset_name : string.
        The prefix of output file.
    out_dir : string, optional
        The path to store the output data.
    different_source : boolean, optional
        True for single cell and bulk data from the same sample, which means not executing batch effect.
        False for single cell and bulk data from the different samples, which means executing batch effect.
    cell_list : list, optional
        The list indicate the cell type names which need to take into consideration.
    scale_factors : int, optional
        The number of counts to normalize every observation to before computing profiles. If `None`, no normalization is performed. 
    trans_method : string, optional
        What transformation to apply to the expression before computing the profiles. 
        - "log" : log(x+1)
        - `None` : no transformation
    save : boolean, optional
        Whether save the result data during each step. If saving, the processing may be skipped.
    save_figure : boolean, optional
        Whether save figures during preprocessing. eg. scatter plot of pca data.
    n_cell : int, optional
        The number of cells within each bulk.
    **kwargs : 
        Additional keyword arguments forwarded to
        :func:`~cytobulk.preprocessing.qc_bulk_sc`.

        
    Returns
    -------
    Returns the deconvolution result and reconstructed st.
    """
    # check the filtered dataset. If exist, skipping preprocessing.
    st_ori_adata = st_adata.copy()
    if exists(f'{out_dir}/filtered/pseudo_bulk_{dataset_name}.h5ad') and exists(f'{out_dir}/filtered/sc_data_{dataset_name}.h5ad') and \
    exists(f'{out_dir}/filtered/bulk_data_{dataset_name}.h5ad') and exists(f'{out_dir}/filtered/cells_{dataset_name}.json'):
        pseudo_st = sc.read_h5ad(f"{out_dir}/filtered/pseudo_bulk_{dataset_name}.h5ad")
        sc_adata = sc.read_h5ad(f"{out_dir}/filtered/sc_data_{dataset_name}.h5ad")
        st_adata = sc.read_h5ad(f"{out_dir}/filtered/bulk_data_{dataset_name}.h5ad")
        with open(f"{out_dir}/filtered/cells_{dataset_name}.json") as json_file:
            common_cell = json.load(json_file)
        annotation_key="curated_cell_type"
    else:
        #preprocessing
        sc_adata, pseudo_st, st_adata,common_cell,annotation_key = pp.preprocessing(st_adata,
                                                                        sc_adata,
                                                                        annotation_key,
                                                                        is_st=True,
                                                                        marker_list=marker_list,
                                                                        rename=rename,
                                                                        dataset_name=dataset_name,
                                                                        out_dir=out_dir,
                                                                        different_source=different_source,
                                                                        cell_list=cell_list,
                                                                        scale_factors=scale_factors,
                                                                        trans_method=trans_method,
                                                                        n_sample_each_group=len(st_adata.obs_names)*6,
                                                                        min_cells_each_group=n_cell,
                                                                        cell_gap_each_group=1,
                                                                        group_number=5,
                                                                        save = save,
                                                                        save_figure=save_figure,
                                                                        **kwargs)
    #deconvolution
    if exists(f'{out_dir}/output/{dataset_name}_prediction_frac.csv'):
        deconv_result = pd.read_csv(f'{out_dir}/output/{dataset_name}_prediction_frac.csv',index_col=0)
    else:
        deconv_result = _bulk_sc_deconv(st_adata, 
                                        pseudo_st, 
                                        sc_adata, 
                                        common_cell,
                                        annotation_key = annotation_key, 
                                        dataset_name=dataset_name, 
                                        out_dir=out_dir,
                                        batch_effect=different_source,
                                        is_st=True)
    threshold = 1 / max_cells
    deconv_result[deconv_result < threshold] = 0
    row_sums = deconv_result.sum(axis=1)
    df_normalized = deconv_result.div(row_sums, axis=0)
    st_ori_adata.uns['deconv']=df_normalized
    df_normalized.to_csv(f"{out_dir}/output/{dataset_name}_prediction_frac_normalized.csv")
    st_ori_adata.write_h5ad(f'{out_dir}/output/{dataset_name}_st_adata.h5ad')
    return df_normalized,st_ori_adata
