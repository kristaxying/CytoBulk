
import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix, csc_matrix
from sklearn.utils.sparsefuncs import inplace_column_scale, inplace_row_scale
import numba
import mkl
import sys
import os
import scipy.sparse
import joblib
from sklearn.decomposition import PCA
from numba import njit, prange
from scipy.optimize import nnls
import multiprocessing
from ._read_data import check_paths

def compute_cluster_averages(adata, annotation_key, common_cell,use_raw=False,project='',save=False,out_dir='./'):
    """
    Compute average expression of each gene in each cluster

    Parameters
    ----------
    adata
        AnnData object of reference single-cell dataset
    annotation_key
        Name of adata.obs column containing cluster labels
    common_cell
        List to store the cell type order.

    Returns
    -------
    pd.DataFrame of cell type average expression of each gene

    """


    if not use_raw:
        x = adata.X
        var_names = adata.var_names
    else:
        if not adata.raw:
            raise ValueError("AnnData object has no raw data, change `use_raw=True, layer=None` or fix your object")
        x = adata.raw.X
        var_names = adata.raw.var_names

    averages_mat = np.zeros((1, x.shape[1]))

    for c in common_cell:
        sparse_subset = csr_matrix(x[np.isin(adata.obs[annotation_key], c), :])
        aver = sparse_subset.mean(0)
        averages_mat = np.concatenate((averages_mat, aver))
    averages_mat = averages_mat[1:, :].T
    df = pd.DataFrame(data=averages_mat, index=var_names, columns=common_cell)
    
    if save:
    # check out path
        reference_out_dir = check_paths(out_dir+'/reference_bulk_data')
        print('Saving average expression data')
        df.to_csv(reference_out_dir+f"/{project}_average_celltype_exp.txt",sep='\t')

    return df


def compute_bulk_with_average_exp(pseudo_bulk, average_cell_exp,save=False,out_dir='./',project=''):
    """
    Compute average expression of each gene in each cluster

    Parameters
    ----------
    pseudo_bulk
        AnnData object of reference single-cell dataset
    annotation_key
        Name of adata.obs column containing cluster labels
    common_cell
        List to store the cell type order.

    Returns
    -------
    pd.DataFrame of cell type average expression of each gene

    """
    pseudo_prop = pseudo_bulk.obs
    cell_sort = average_cell_exp.columns
    # resort prop data
    pseudo_prop = pseudo_prop[cell_sort]
    # truth bulk
    sample = np.zeros((pseudo_prop.shape[0],average_cell_exp.shape[0]))
    for i in range(pseudo_prop.shape[0]):
        cell_exp = pseudo_prop.iloc[i,:] * average_cell_exp
        sample[i] = cell_exp.sum(axis=1)
    # format dataframe
    sample = pd.DataFrame(sample,index=pseudo_bulk.obs_names,columns=pseudo_bulk.var_names)
    if save:
        reference_out_dir = check_paths(out_dir+'/reference_bulk_data')
        sample.to_csv(reference_out_dir+f"/{project}_cell_type_ave_bulk.txt",sep='\t')

    return sample

def data_dict_integration(data_df,data_dict,common_cell,top_num=100):
    """
    Integration the dataframe and dict.

    Parameters
    ----------
    data
        Dataframe and the columns is the key.
    dict
        Dictionary with key and values.
    top_number
        The top highest value will be keeped.

    Returns
    -------
    pd.DataFrame of cell type average expression of each gene

    """
    key_list = data_dict.keys()
    overlapping_gene=[]
    for i in key_list:
        tmp_index = data_df[i].sort_values().iloc[:top_num].keys()
        common_index = np.union1d(np.array(tmp_index), np.array(data_dict[i]))
        if len(common_index)>0:
            data_dict[i] = list(common_index)
            overlapping_gene += data_dict[i]
        else:
            del data_dict[i]
            common_cell.remove(i)
    overlapping_gene = list(set(overlapping_gene))
    return data_dict, overlapping_gene, common_cell




