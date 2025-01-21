import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from ._read_data import check_paths

def compute_cluster_averages(adata, annotation_key, common_cell,use_raw=False,project='',save=False,out_dir='./'):
    """
    Compute average expression of each gene in each cluster.

    Parameters
    ----------
    adata
        AnnData object of reference single-cell dataset.
    annotation_key
        Name of adata.obs column containing cluster labels.
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

def data_dict_integration(common_list,data_df,data_dict,common_cell,top_num=100):
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
    pd.DataFrame of cell type average expression of each gene.

    """
    key_list = data_dict.keys()
    overlapping_gene=[]
    for i in key_list:
        # find marker gene name in each cell type
        tmp_index = data_df[i].sort_values().iloc[:top_num].keys()
        if data_dict[i]!=[]:
            common_index = np.union1d(np.array(tmp_index), np.array(data_dict[i]))
        else:
            common_index = np.array(tmp_index)
        common_index = np.intersect1d(np.array(common_list),common_index)
        if len(common_index)>0:
            data_dict[i] = list(common_index)
            overlapping_gene += data_dict[i]
        else:
            del data_dict[i]
            common_cell.remove(i)
    overlapping_gene = np.unique(np.array(overlapping_gene))
    return data_dict, overlapping_gene, common_cell

def marker_integration(common_list,data_df,marker_list):
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
    pd.DataFrame of cell type average expression of each gene.

    """

    overlapping_gene=[]
    data_df["mean"] = data_df.mean(axis=1)
    for i in data_df.columns:
        # find marker gene name in each cell type
        tmp_index = data_df.loc[data_df[i]>data_df["mean"]].index.tolist()
        overlapping_gene += tmp_index
    if len(marker_list)>0:   
        common_gene = np.union1d(overlapping_gene, marker_list)
    else:
        common_gene=np.array(overlapping_gene)
    common_gene = np.intersect1d(np.array(common_list),common_gene)

    return common_gene
    '''
    key_list = data_dict.keys()
    overlapping_gene=[]
    data_df["mean"] = data_df.mean(axis=1)
    for i in key_list:
        # find marker gene name in each cell type
        tmp_index = data_df.loc[data_df[i]>data_df["mean"]].index.tolist()
        if len(tmp_index)!=0:
            if len(data_dict[i])!=0:
                common_index = np.union1d(np.array(tmp_index), np.array(data_dict[i]))
            else:
                common_index = np.array(tmp_index)
        else:
            common_index = np.array(data_dict[i])
        common_index = np.intersect1d(np.array(common_list),common_index)
        if len(common_index)>0:
            data_dict[i] = list(common_index)
            overlapping_gene += data_dict[i]
        else:
            del data_dict[i]
            common_cell.remove(i)
    overlapping_gene = np.unique(np.array(overlapping_gene))
    return data_dict, overlapping_gene, common_cell
    '''

def filter_samples(pseudo_bulk, bulk_adata,data_num,num=20,cut_off=0.9,loc=None):
    from sklearn.metrics.pairwise import cosine_similarity
    sample_name = pseudo_bulk.obs_names
    if loc:
        bulk_matrix = bulk_adata.layers[loc]
        pseudo_matrix = pseudo_bulk.layers[loc]
    else:
        bulk_matrix = bulk_adata.X
        pseudo_matrix = pseudo_bulk.X
    #similarity_matrix = np.matmul(bulk_matrix,pseudo_matrix.T)
    similarity_matrix = cosine_similarity(bulk_matrix,pseudo_matrix)
    top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, :num]
    selected_indices = np.unique(top_k_indices.flatten())
    if 5*similarity_matrix.shape[0]<2000:
        target_num = 2000
    else:
        target_num=5*similarity_matrix.shape[0]
    while len(selected_indices)<target_num:
          num=int(num*1.5)
          top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, :num]
          selected_indices = np.unique(top_k_indices.flatten())
    return sample_name[selected_indices]

def compute_average_cosin(pseudo_bulk_tensor,bulk_adata,loc=None):
    from sklearn.metrics.pairwise import cosine_similarity
    valid_list=[]
    for i in pseudo_bulk_tensor:
        valid_list.append(i[0].numpy())
    pseudo_matrix = np.array(valid_list)
    if loc:
        bulk_matrix = bulk_adata.layers[loc]
    else:
        bulk_matrix = bulk_adata.X
    sample_cosine = cosine_similarity(pseudo_matrix,bulk_matrix)
    cosin_max = np.mean(sample_cosine, axis=1)
    return np.average(cosin_max)


def filter_gene(expression,reference,out_dir,cell_type,save=True):
    data_list=[expression,reference]
    filtered_list=[]
    for df in data_list:
        zero_count_per_column = (df == 0).sum(axis=0)
        columns_to_keep = zero_count_per_column[zero_count_per_column <= (len(df) / 2)].index
        df_filtered = df[columns_to_keep]
        filtered_list.append(df_filtered)
    common_columns = filtered_list[0].columns.intersection(filtered_list[1].columns)
    if save==True:
        reference_out_dir = check_paths(out_dir+'/test_data')
        reference[common_columns].to_csv(reference_out_dir+f"/{cell_type}.txt",sep='\t')
        

    return expression[common_columns],reference[common_columns]



def normalize_data(data):
    data = np.nan_to_num(data).astype(float)
    data *= 10**6 / np.sum(data, axis=0, dtype=float)
    np.log2(data + 1, out=data)
    np.nan_to_num(data, copy=False)
    return data