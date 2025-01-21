import pandas as pd
import anndata as ad
import numpy as np
import scanpy as sc
from scipy.sparse import issparse,csr_matrix
from .. import utils
import time


#human_sc min_counts_per_cell=50 min_genes_per_cell=100
#hnsc min_counts_per_cell=1 min_genes_per_cell=5
def _filter_adata(
    adata,
    min_counts_per_gene=None,
    min_counts_per_cell=10,
    min_cells_per_gene=1,
    min_genes_per_cell=100,
    remove_constant_genes=True,
    remove_zero_cells=True,
    save=False,
    out_dir='.',
    project='',
    ignore_duplicated_genes=True
    ):

    """
    Filter one or more :class:`~anndata.AnnData` to satisfy simple quality criteria. 

    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` containing the expression to filter. 
    min_counts_per_gene: int, optional
        The minimum count (per :class:`~anndata.AnnData`) genes must have to be kept.
    min_counts_per_cell: int, optional
        The minimum count (per :class:`~anndata.AnnData`) cells must have to be kept.
    min_cells_per_gene: int, optional
        The minimum number of cells (per :class:`~anndata.AnnData`) genes must have to be kept.
    min_genes_per_cell: int, optional
        The minimum number of genes (per :class:`~anndata.AnnData`) cells must have to be kept.
    remove_constant_genes: boolean, optional
        Whether to remove genes which do not show any variation between cells
    remove_zero_cells: boolean, optional
        Whether to remove cells without non-zero genes.
        
    Returns
    -------
    Returns the filtered `adata`.
    
    """

    #check gene
    if not ignore_duplicated_genes:
        start_t = time.perf_counter()
        if not isinstance(adata.X, np.ndarray):
            adata.X = adata.X.toarray()
        print("Drop duplicated genes")
        #generate new data, average duplicated genes
        df_data = pd.DataFrame(adata.X,index=adata.obs_names,columns=adata.var_names).transpose()
        df_data['GeneSymbol'] = df_data.index
        df_data['GeneSymbol'] = df_data['GeneSymbol'].apply(lambda x: x.split('.')[0])
        df_data = df_data.groupby('GeneSymbol').mean()
        df_data = df_data.reset_index()
        df_data = pd.DataFrame(df_data.values[:,1:],index=df_data['GeneSymbol'].values,columns=df_data.columns[1:])
        meta = adata.obs
        cell_id = meta.index.tolist()
        common_id = list(set(cell_id) & set(df_data.columns))
        df_data = df_data.loc[:,common_id]
        meta = meta.loc[common_id,:]
        meta = meta.loc[df_data.columns,:]
        print(f"Found {df_data.shape[0]} not duplicated genes")
        X = np.array(df_data.values.T,dtype=float)
        X = csr_matrix(X)
        adata = sc.AnnData(X=X,obs=meta)
        adata.var_names = df_data.index
        print(f'Time to check the gene name and average the expression over duplicated genes: {round(time.perf_counter() - start_t, 2)} seconds')
    
    print("Start filtering")
    start_t = time.perf_counter()
    valid_genes = adata.var.index
    if (min_counts_per_gene is None and min_cells_per_gene is None and not remove_constant_genes) or np.prod(adata.shape) == 0:
        valid_genes = valid_genes.intersection(adata.var.index)
    else:
        if min_counts_per_gene is not None:
            valid_genes = valid_genes.intersection(adata.var.index[sc.pp.filter_genes(adata, min_counts=min_counts_per_gene, inplace=False)[0]])
        if min_cells_per_gene is not None:
            valid_genes = valid_genes.intersection(adata.var.index[sc.pp.filter_genes(adata, min_cells=min_cells_per_gene, inplace=False)[0]]) 
        if remove_constant_genes:
            if len(adata.obs) == 1:
                print('WARNING: Encountered a dataset with only a single observation while `remove_constant_genes` is enabled! As this would remove all genes, `remove_constant_genes` is disabled for this dataset.')
            else:
                not_constant = adata.X.max(axis=0)!=adata.X.min(axis=0)
                if issparse(not_constant):
                    not_constant = not_constant.todense().A
                not_constant = not_constant.flatten()
                valid_genes = valid_genes.intersection(adata.var.index[not_constant])
    if len(adata.var.index) != len(valid_genes): # filter happened
                adata = adata[:,valid_genes]
    elif (adata.var.index != valid_genes).any(): # reordering happened: no side effects on cell filtering
                adata = adata[:,valid_genes]
    ## check sample
    if np.prod(adata.shape) == 0:
        raise ValueError('Encountered a dataset with no high quality genes')

    cell_mask = np.full(len(adata.obs.index),True)
    if remove_zero_cells:
        cell_mask &= utils.get_sum(abs(adata.X),axis=1) != 0
    if min_counts_per_cell is not None:
        cell_mask &= sc.pp.filter_cells(adata, min_counts=min_counts_per_cell, inplace=False)[0]
    if min_genes_per_cell is not None:
        cell_mask &= sc.pp.filter_cells(adata, min_genes=min_genes_per_cell, inplace=False)[0]
    
    if not cell_mask.all():
        adata = adata[cell_mask]
    if np.prod(adata.shape) == 0:
        raise ValueError('Encountered a dataset with no valid samples or cells')
    
    print(f'The number of valid samples is {adata.shape[0]}, the number of valid genes is {adata.shape[1]}')

    print(f'Time to find subset samples which meet quality criterion: {round(time.perf_counter() - start_t, 2)} seconds')

    if save:
        out_dir = utils.check_paths(f'{out_dir}/filtered')
        adata.write_h5ad(f"{out_dir}/filtered_{project}.h5ad") 
    return adata


def _filter_dataframe(
    dataframe,
    min_counts_per_gene=None,
    min_counts_per_sample=100,
    min_cells_per_gene=None,
    min_genes_per_sample=100,
    remove_constant_genes=True,
    remove_zero_sample=True,
    project='',
    out_dir='.',
    **kwargs
    ):

    """
    Filter one or more :class:~pandas.dataframe to satisfy simple quality criteria. 

    Parameters
    ----------
    dataframe
        An :class:`~pandas.dataframe` containing the expression to filter. 
    min_counts_per_gene: int, optional
        The minimum count (per :class:`~pandas.dataframe`): genes must have to be kept.
    min_counts_per_sample: int, optional
        The minimum count (per :class:`~pandas.dataframe`): samples must have to be kept.
    min_samples_per_gene: int, optional
        The minimum number of samples (per :class:`~pandas.dataframe`): genes must have to be kept.
    min_genes_per_sample: int, optional
        The minimum number of genes (per :class:`~pandas.dataframe`) samplse must have to be kept.
    remove_constant_genes: boolean, optional
        Whether to remove genes which do not show any variation between samples.
    remove_zero_cells: boolean, optional: boolean, optional
        Whether to remove sample without non-zero genes.

    Returns
    -------
    Returns the filtered `adata`.
    
    """
    ## convert to `~anndata.AnnData`
    adata = sc.AnnData(dataframe)
    adata = _filter_adata(adata,
                        min_counts_per_gene=min_counts_per_gene,
                        min_counts_per_cell=min_counts_per_sample,
                        min_cells_per_gene=min_cells_per_gene,
                        min_genes_per_cell=min_genes_per_sample,
                        remove_constant_genes=remove_constant_genes,
                        remove_zero_cells=remove_zero_sample,
                        project=project,
                        out_dir=out_dir,
                        **kwargs)
    
    return adata

def qc_bulk_sc(bulk_data,
            sc_adata,
            min_counts_per_sample=50,
            min_genes_per_sample=100,
            out_dir='.',
            dataset_name='',
            **kwargs):
    """
    Quality control in bulk dataframe and sc adata.

    Parameters
    ----------
    bulk_data: dataframe
        An :class:`~pandas.dataframe` containing the bulk expression data. 
        The first column should be gene symbol, following column should be sample name.
    sc_adata: anndata.AnnData
        An :class:`~anndata.AnnData` containing the expression to filter.
    min_counts_per_sample: int, optional
        The minimum count (per :class:`~pandas.dataframe`): samples must have to be kept.
    min_genes_per_sample: int, optional
        The minimum number of genes (per :class:`~pandas.dataframe`) samplse must have to be kept.
    **kwargs:
        Additional keyword arguments forwarded to
        :func:`~cytobulk.preprocessing._filter_adata`.
    Returns
    -------
    Returns the filtered bulk data (adata) and sc data (adata).

    """
    sc_adata.var_names_make_unique()
    
    project_sc = dataset_name+'_sc'
    project_bulk = dataset_name+'_bulk'
    # check data format
    if not isinstance(sc_adata,ad.AnnData):
        raise ValueError(f"`A` can only be a dataframe but is a {sc_adata.__class__}!")
    if isinstance(bulk_data,pd.DataFrame):
        print("------------------start filtering bulk data-------------------------")
        filtered_bulk = _filter_dataframe(bulk_data,
                                        min_counts_per_sample=min_counts_per_sample,
                                        min_genes_per_sample=min_genes_per_sample,
                                        project = project_bulk,
                                        out_dir=out_dir,
                                        **kwargs)
        print("------------------finish filtering bulk data-------------------------")
    else:
        bulk_data.var_names_make_unique()
        filtered_bulk = _filter_adata(bulk_data,
                                    min_counts_per_cell=min_counts_per_sample,
                                    min_genes_per_cell=min_genes_per_sample,
                                        project = project_bulk,
                                        out_dir=out_dir,
                                        **kwargs)

    print("------------------start filtering single cell data-------------------------")
    filtered_sc = _filter_adata(sc_adata,
                                project = project_sc,
                                out_dir=out_dir,
                                **kwargs)
    print("------------------finish filtering single cell data-------------------------")
    # get overlapping gene between sc adata and bulk adata
    common_gene = filtered_bulk.var.index.intersection(filtered_sc.var.index)
    # subset datasets
    filtered_sc = filtered_sc[:,common_gene]
    filtered_bulk = filtered_bulk[:,common_gene]
    print("finish getting overlapping genes")
    return filtered_sc, filtered_bulk, common_gene

def high_variable_gene(adata,flavor):
    sc.pp.highly_variable_genes(adata, min_mean=0, max_mean=np.inf, min_disp=0.25,flavor=flavor)
    print("length")
    print(len(adata.var.highly_variable))
    if len(adata.var.highly_variable)>4000:
        sc.pp.highly_variable_genes(adata, min_mean=0, max_mean=np.inf, min_disp=0.5,flavor=flavor)
    if len(adata.var.highly_variable)>4000:
        sc.pp.highly_variable_genes(adata, n_top_genes=4000)
    return adata[:, adata.var.highly_variable]

def qc_st_sc(st_adata,
            sc_adata,
            min_counts_per_sample=1,
            min_genes_per_sample=5,
            dataset_name='',
            out_dir='.',
            **kwargs):
    """
    Quality control in bulk dataframe and sc adata.

    Parameters
    ----------
    bulk_data: dataframe
        An :class:`~pandas.dataframe` containing the bulk expression data. 
        The first column should be gene symbol, following column should be sample name.
    sc_adata: anndata.AnnData
        An :class:`~anndata.AnnData` containing the expression to filter.
    min_counts_per_sample: int, optional
        The minimum count (per :class:`~pandas.dataframe`): samples must have to be kept.
    min_genes_per_sample: int, optional
        The minimum number of genes (per :class:`~pandas.dataframe`) samplse must have to be kept.
    **kwargs:
        Additional keyword arguments forwarded to
        :func:`~cytobulk.preprocessing._filter_adata`.
    Returns
    -------
    Returns the filtered bulk data (adata) and sc data (adata).

    """
    sc_adata.var_names_make_unique()
    st_adata.var_names_make_unique()
    project_sc = dataset_name+'_sc'
    project_st = dataset_name+'_st'
    # check data format
    if not isinstance(sc_adata,ad.AnnData):
        raise ValueError(f"`A` can only be a dataframe but is a {sc_adata.__class__}!")
    if not isinstance(st_adata,ad.AnnData):
        raise ValueError(f"`A` can only be a anndata.AnnData but is a {st_adata.__class__}!")
    print("------------------start filtering st data-------------------------")
    filtered_st = _filter_adata(st_adata,
                                    min_counts_per_cell=min_counts_per_sample,
                                    min_genes_per_cell=min_genes_per_sample,
                                    min_cells_per_gene=10,
                                    project = project_st,
                                    out_dir= out_dir,
                                    **kwargs)
    print("------------------finish filtering st data-------------------------")

    print("------------------start filtering single cell data-------------------------")
    filtered_sc = _filter_adata(sc_adata,
                                project = project_sc,
                                out_dir=out_dir,
                                **kwargs)
    print("------------------finish filtering single cell data-------------------------")
    # get overlapping gene between sc adata and bulk adata
    common_gene = filtered_st.var.index.intersection(filtered_sc.var.index)
    # subset datasets
    filtered_sc = filtered_sc[:,common_gene]
    filtered_st = filtered_st[:,common_gene]
    print(f"finish getting {len(common_gene)} overlapping genes")
    return filtered_sc, filtered_st, common_gene

def qc_sc(sc_adata,**kwargs):
    """
    Quality control in sc adata.

    Parameters
    ----------
    sc_adata: anndata.AnnData
        An :class:`~anndata.AnnData` containing the expression to filter.
    **kwargs:
        Additional keyword arguments forwarded to
        :func:`~cytobulk.preprocessing._filter_adata`.
    Returns
    -------
    Returns the filtered sc data (adata).

    """

    if not isinstance(sc_adata,ad.AnnData):
        raise ValueError(f"`A` can only be a ad.AnnData but is a {sc_adata.__class__}!")
    print("------------------start filtering single cell data-------------------------")
    filtered_adata = _filter_adata(sc_adata,**kwargs)
    print("------------------finish filtering single cell data-------------------------")
    return filtered_adata