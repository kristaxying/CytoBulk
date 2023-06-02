import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import sys
import warnings
from utils import *
import anndata
import scanpy as sc
from pathlib import Path
import os
import warnings


def check_paths(output_folder,output_prefix=None):
    # Create relative path
    output_path = os.path.join(os.getcwd(), output_folder)

    # Make sure that the folder exists
    Path(output_path).mkdir(parents=True, exist_ok=True)

    if os.path.exists(os.path.join(output_path, f"{output_prefix}assigned_locations.csv")):
        print("\033[91mWARNING\033[0m: Running this will overwrite previous results, choose a new"
              " 'output_folder' or 'output_prefix'")

    return output_path



def read_file(file_path, file_label):
    '''
    Read data with given path.
    args:
        file_path: file path.
        file_label: the file label to raise exception.
    return:
        the read file.
    '''
    try:
        file_delim = "," if file_path.endswith(".csv") else "\t"
        with warnings.catch_warnings():
            file_data = pd.read_csv(file_path, sep=file_delim).dropna()
    except Exception as e:
        raise IOError (f"Make sure you provided the correct path to {file_label} files. "
                      "The following input file formats are supported: .csv with comma ',' as "
                      "delimiter, .txt or .tsv with tab '\\t' as delimiter.")
    return file_data


def read_marker_genes(marker_path):
    '''
    Get the maker genes for different cell types.
    args:
        file_path: marker gene file path.
    return:
        the list of marker_genes_by_cell_type.
    '''
    marker_genes = read_file(marker_path,'marker_gene')
    return marker_genes.to_dict('list')


def read_training_data(sc_path,meta_path,marker,sc_nor,out_dir,save_ref=True):

    """read sc data and meta information to train model.
    args:
        sc_path:    sc-rna data path.
        meta_path:  meta data with cell type information path.
        marker:     the marker gene list if provided or none.
        sc_nor:     Boolean, true for using preprocessing on sc data.
        out_path:   the dir to store the result files.
    """
    warnings.filterwarnings(action='ignore', category=FutureWarning) 
    warnings.filterwarnings(action='ignore', category=UserWarning)
    warnings.filterwarnings(action='ignore', category=RuntimeWarning)
    # read data
    print("Start to read training data and check the data format...")
    if sc_path.endswith(".h5ad"):
        sc_data = sc.read_h5ad(sc_path)
        meta = pd.DataFrame(sc_data.obs,index=sc_data.obs_names)
        meta = meta.loc[sc_data.obs_names,:]
        meta.index.name = 'Cell'
        if isinstance(sc_data.X, np.ndarray):
            pass
        else:
            sc_data.X = sc_data.X.toarray()
    elif (sc_path is None) and (meta_path is None):
        raise ValueError("For sc data, you must provide either a .h5ad file or paths for expression and meta.")
    else:
        sc_data = read_file(sc_path,'training scRNA-seq')
        
        meta = read_file(meta_path,'training label')
        sc_data = anndata.AnnData(sc_data)
    if 'Celltype_major' or 'Celltype_minor' not in meta.columns:
        TypeError("The metadata need to have 'Celltype_major' and 'Celltype_minor' information")

    ## if marker_label = 'auto_find', find marker gene firstly.
    if marker == None and sc_nor:
        print("Start to preprocess the ref scRNA seq data...")
        print("Filtering cells...")
        sc.pp.filter_cells(sc_data, min_genes=200)
        sc.pp.filter_genes(sc_data, min_cells=10)
        sc_data.var['mt'] = sc_data.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
        sc.pp.calculate_qc_metrics(sc_data, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        sc_data = sc_data[sc_data.obs.n_genes_by_counts < 2500, :]
        sc_data = sc_data[sc_data.obs.pct_counts_mt < 5, :]
        sc_data.raw = sc_data
        print("Normalizing count...")
        sc.pp.normalize_total(sc_data, target_sum=1e4)
        sc.pp.log1p(sc_data)
        sc.pp.highly_variable_genes(sc_data, min_mean=0.0125, max_mean=3, min_disp=0.5)
        print("Finding marker genes...")
        sc.tl.rank_genes_groups(sc_data, 'Celltype_minor', method='t-test')
        marker = pd.DataFrame(sc_data.uns['rank_genes_groups']['names']).head(25)
        print("Saving marker genes...")
        marker.to_csv(out_dir+"/marker_gene.csv")
        marker = marker.to_dict('list')
        marker_cell = list(marker.keys())

    # check the cell types of makrer genes and meta data are matched.
    else:
        marker_cell = list(marker.keys())
        common_cell = set(meta) & set(marker_cell)
        if (len(common_cell) != len(meta)) or (len(common_cell) != len(marker_cell)):
            raise ValueError(f"In training, the meta data and self definded marker genes have to "
                         "have the cell type,please check your file ")
    # format data
    sc_data = pd.DataFrame(sc_data.X,index=sc_data.obs_names,columns=sc_data.var_names).transpose()
    sc_data.index.name = 'GeneSymbol'
    cell_id = meta.index.tolist()
    common_id = list(set(cell_id) & set(sc_data.columns))
    sc_data = sc_data.loc[:,common_id]
    meta = meta.loc[common_id,:]
    print(f"Found {len(common_id)} valid cell")
    # save ref scrnaseq data
    out_dir = check_paths(out_dir+'/cell_feature')
    print('Saving ref scrna seq data...')
    save_ref = False
    if save_ref:
        for i in marker_cell: 
            cell_id = meta[meta['Celltype_minor']==i].index.tolist()
            ref_sc = sc_data.loc[:,cell_id]
            ref_sc.to_csv(out_dir+f"/{i}_scrna.txt",sep='\t')

    return sc_data, meta, marker


def read_bulk_data(bulk_exp_path,marker_genes_by_cell_type):
    """read bulk rna data and check the dataformat.
    args:
        bulk_exp_path: path of bulk rna data.
        marker_genes_by_cell_type: the marker gene list.
    """
    bulk_data = read_file(bulk_exp_path,"bulk rna")
    ## check format
    if 'GeneSymbol' not in bulk_data.columns:
        ValueError("There is no column named 'GeneSymbol'.")
    duplicates = check_duplicates(input_genes)
    if duplicates:
        ValueError(f"Duplicated input genes found, please check your data")
    unaccountable_cell_type = find_unaccountable_cell_types(marker_genes_by_cell_type, bulk_exp.index)
    if unaccountable_cell_type:
        ValueError(f"The input genes don't account for at least half of the marker genes in the cell type: {unaccountable_cell_type}")
    else:
        print("All cell types have at least half of their genes accounted for.")

    return bulk_data





