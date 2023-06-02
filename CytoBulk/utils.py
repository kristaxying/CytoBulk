import pandas as pd
import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix
import sys
import warnings
from read_data import *
from tqdm import tqdm
import time
import os
from numpy.random import choice

def check_paths(output_folder,output_prefix=None):
    # Create relative path
    output_path = os.path.join(os.getcwd(), output_folder)

    # Make sure that the folder exists
    Path(output_path).mkdir(parents=True, exist_ok=True)

    if os.path.exists(os.path.join(output_path, f"{output_prefix}assigned_locations.csv")):
        print("\033[91mWARNING\033[0m: Running this will overwrite previous results, choose a new"
              " 'output_folder' or 'output_prefix'")

    return output_path


###########################################
# read bulk rna data and check the format
###########################################
def find_unaccountable_cell_types(marker_genes_by_cell_type, input_genes):

    """check the whether the input gene >= half of marker gene for each cell type or not.
    args:
        marker_genes_by_cell_type:  the marker gene list.
        input_genes:                input gene list need to be checked.
    """

    for cell_type, marker_genes in marker_genes_by_cell_type.items():
        intersect_count = len(set(input_genes).intersection(marker_genes))
        if intersect_count < len(marker_genes) / 2:
            return cell_type
    return None

def check_duplicates(input_genes):
    """check whether there has duplicated genes.
    args:
        input_genes: input gene list need to be checked.
    """
    seen_genes = set()
    for gene in input_genes:
        if gene in seen_genes:
            return gene
        seen_genes.add(gene)
    return None

def get_stimulation(n_celltype,n_sample,meta,sc_data,out_dir,n,type='training'):
    """get stimulated expression data and cell type prop.
    args:
        type: string, training or testing.
    """
    print(f'The {type} data generation...')
    print("start to generate cell fraction...")
    cell_prop = np.random.dirichlet(np.ones(n_celltype), n_sample)
    # get cell meta dictionary
    meta_index = meta[['Celltype_minor']]
    meta_index = meta_index.groupby(meta['Celltype_minor']).groups
    for key, value in meta_index.items():
        meta_index[key] = np.array(value)
    cell_prop = cell_prop / np.sum(cell_prop, axis=1).reshape(-1, 1)
    print(f'The number of samples is {cell_prop.shape[0]}, the number of cell types is {cell_prop.shape[1]}')
    for i in range(int(cell_prop.shape[1] * 0.1)):
        indices = np.random.choice(np.arange(cell_prop.shape[0]), replace=False, size=int(cell_prop.shape[0] * 0.1))
        cell_prop[i, indices] = 0
    # get cell number based on cell prop
    print('Start sampling...')
    sample = np.zeros((cell_prop.shape[0],sc_data.shape[0]))
    allcellname = meta_index.keys()
    cell_prop = cell_prop / np.sum(cell_prop, axis=1).reshape(-1, 1)
    cell_num = np.floor(n * cell_prop)
    for i, sample_prop in tqdm(enumerate(cell_num)):
        for j, cellname in enumerate(allcellname):
            select_index = choice(meta_index[cellname], size=int(sample_prop[j]), replace=True)
            sample[i] += sc_data.loc[:,select_index].sum(axis=1)
    sample = sample/n
    print("Sampling down")
    # print("Saving simulated data...")
    #out_dir = check_paths(out_dir+'/training_data')
    sample = pd.DataFrame(sample,index=['Sample'+str(i) for i in range(n_sample)],columns=sc_data.index.values)
    # sample.to_csv(out_dir+"/expression.csv")
    cell_prop = pd.DataFrame(cell_prop,index=['Sample'+str(i) for i in range(n_sample)],columns=allcellname)
    # cell_prop.to_csv(out_dir+"/fraction.csv")
    return sample,cell_prop



def bulk_simulation(sc_path,meta_path,marker,sc_nor,out_dir,n_sample=5000,n=200,test=True):
    """get stimulated training data and testing data.
    args:
        test: boolean, true for generating testing data.
    """
    # read data
    start_t = time.perf_counter()
    sc_data, meta, marker = read_training_data(sc_path, meta_path, marker, sc_nor, out_dir)
    print(f"Time to read and check training data: {round(time.perf_counter() - start_t, 2)} seconds")

    # training bulk rna
    train_out_dir = check_paths(out_dir+'/training_data')
    print('====================================================================')
    print('====================================================================')
    print('Start to stimulate the training bulk expression data ...')
    start_t = time.perf_counter()
    n_celltype = len(meta['Celltype_minor'].value_counts())
    training_data,training_prop = get_stimulation(n_celltype,n_sample,meta,sc_data,train_out_dir,n,type='training')
    print(f'Time to generate training data: {round(time.perf_counter() - start_t, 2)} seconds')

    # testing bulk rna
    if test:
        print('-----------------------------------------------------------------')
        print('start to stimulate the testing bulk expression data ...')
        testing_out_dir = check_paths(out_dir+'/testing_data')
        start_t = time.perf_counter()
        testing_data,testing_prop = get_stimulation(n_celltype,int(n_sample*0.02),meta,sc_data,testing_out_dir,n,type='testing')
        print(f'Time to generate the training data: {round(time.perf_counter() - start_t, 2)} seconds')

    return training_data,training_prop,testing_data,testing_prop,marker
    


    #celltype_groups = sc_data.groupby('celltype').groups








##################################
# for bulk rna data normalization
##################################
def read_gene_length_data(file_path):
    df = pd.read_csv(file_path)
    df = df.set_index('SYMBOL')
    df['lengths'] /= 1000  # Convert to kilobases
    return df

def bulk_normalization(bulk_exp,nor_strategy,gene_length_path):
    """
        Get the normalized bulk data.
    args:
        nor_strategy: normalization strategy.
    return:
        normalized bulk data.
    """
    gene_length = read_gene_length_data(gene_length_path)
    exp_col = bulk_exp.columns[1:]
    max_exp_value = bulk_exp[exp_col].max().max()
    if max_exp_value > 100 & nor_strategy=="log2(tpm)":
        matched_gene_lengths = bulk_exp['GeneSymbol'].map(gene_length['lengths'].to_dict())
        # Step 1: Divide read counts by gene length (in kilobases)
        rpkm_data = bulk_exp.iloc[:, 1:].div(matched_gene_lengths, axis=0)
        # Step 2: Normalize sequencing depth and multiply by 1e6 to get TPM values
        tpm_data = rpkm_data.div(rpkm_data.sum(axis=0)) * 1e6
        # Step 3: log2(tpm+1)
        log2_tpm_data = np.log2(tpm_data.iloc[:, 1:] + 1)
        norm_data = pd.concat([bulk_exp['GeneSymbol'], log2_tpm_data], axis=1)

    if max_exp_value > 100 & nor_strategy=="log2":
        log2_tpm_data = np.log2(bulk_exp.iloc[:, 1:] + 1)
        norm_data = pd.concat([bulk_exp['GeneSymbol'], log2_tpm_data], axis=1)
    if max_exp_value < 10:
        ValueError("The max_exp_value of bulk data is smaller than 10, so please use nor_strategy = none")
        
    return norm_data
    




def image_preprocessing(he_image):
    """

    """
        


def st_preprocessing(st_data):
    """

    """



def st_imputation(st_data,image_exp):
    """

    """

def reorignize_data(exp_data,meta_data,normback_exp=None):
    if normback_exp is not None:
        counts = csr_matrix(normback_exp, dtype=int)
    else:
        counts = csr_matrix(exp_data.values, dtype=np.float32)
    adata = ad.AnnData(counts,obs=meta_data)
    adata.var_names =exp_data.columns
    adata.var['SYMBOL'] = adata.var_names
    adata.var['MT_gene'] = [gene.startswith('MT-') for gene in adata.var['SYMBOL']]
    adata.obsm['MT'] = adata[:, adata.var['MT_gene'].values].X.toarray()
    adata = adata[:, ~adata.var['MT_gene'].values]
    adata.raw = adata
    return adata

def norm_back(exp_data):
    normalised_data = np.around((np.exp(exp_data.values.T) - 1),4)
    #normalised_data=('%. 4f' % normalised_data)
    np.savetxt('./test.txt', normalised_data, delimiter='\t')   # X is an array
    #count_per_cell =np.sum(normalised_data,axis=1)
    #data = normalised_data / 10000 * count_per_cell.reshape((normalised_data.shape[0],1))
    #print(data.min())
    #contain_nan = (True in np.isnan(data))
    #print(contain_nan)
    #print(count_per_cell!=count_per_cell)
    data = normalised_data.astype(int)
    return data


    