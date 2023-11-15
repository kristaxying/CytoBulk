
import pandas as pd
import numpy as np
import os
from multiprocessing import Process
from tqdm import tqdm
from .. import get
from .. import utils
import time




def bulk_mapping(frac_data,sc_adata,bulk_adata,n_cell,annotation_key,bulk_layer=None,sc_layer=None,reorder=True,multiprocessing=True):
    start_t = time.perf_counter()
    print("=================================================================================================")
    print('Start to mapping bulk data with single cell dataset.')
    # format data
    cell_prop = frac_data.values
    cell_num = np.floor(n_cell * cell_prop)
    meta_data = sc_adata.obs[[annotation_key]]
    meta_dict = meta_data.groupby(meta_data[annotation_key]).groups
    for key, value in meta_dict.items():
        meta_dict[key] = np.array(value)
    cellname_list=frac_data.columns
    cell_list = sc_adata.obs.index
    
    #normalization
    sc_data = get.count_data(sc_adata,counts_location=sc_layer)
    bulk_data = get.count_data(bulk_adata,counts_location=bulk_layer)
    if reorder:
        gene_order = sc_data.index.tolist()
        bulk_data = bulk_data.loc[gene_order]
    sc_data = utils.normal_center_df(sc_data)
    bulk_data = utils.normal_center_df(bulk_data)
    sample = np.zeros((cell_prop.shape[0],sc_data.shape[0]))
    mapped_cor = []

    #compute person correlation and select sc according to person correlation.
    sc_mapping_dict = dict([(k,[]) for k in bulk_data.columns])
    for i, sample_num in tqdm(enumerate(cell_num)):
        name = bulk_data.columns[i]
        sample_cor = np.dot(bulk_data[name].values.reshape(1,bulk_data.shape[0]),sc_data.values)
        cor_index = cell_list[np.argsort(sample_cor)]
        for j, cellname in enumerate(cellname_list):
            mask = np.isin(cor_index, meta_dict[cellname])
            sub_cell = cor_index[mask]
            sub_cell = sub_cell[:int(sample_num[j])]
            sc_mapping_dict[name].extend(sub_cell)
        sample[i] = sc_data.loc[:,sc_mapping_dict[name]].sum(axis=1)
        mapped_cor.append(utils.pear(sample[i],bulk_data[name].values))

    print('mapping solution:',"min correlation", min(mapped_cor),"average correlation",np.mean(mapped_cor),"max correlation", max(mapped_cor))

    bulk_adata.obsm['cell_fraction'] = frac_data
    bulk_adata.obsm['cell_number']=pd.DataFrame(cell_num,index=frac_data.index,columns=frac_data.columns).transpose()
    bulk_adata.layers['mapping'] = sample

    print(f'Time to finish mapping: {round(time.perf_counter() - start_t, 2)} seconds')
    print("=================================================================================================")

    return bulk_adata,sc_mapping_dict