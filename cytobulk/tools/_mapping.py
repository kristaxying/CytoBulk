
import pandas as pd
import numpy as np
import os
import multiprocessing as mp
from tqdm import tqdm
from .. import get
from .. import utils
import time
from multiprocessing import Pool, cpu_count
import numpy as np
from ._cytospace import main_cytospace

def _bulk_mapping_parallel(i, cell_num, bulk_data, sc_data, cell_list, meta_dict, cellname_list, original_sc):
    """
    Parallel function for bulk reconstruction.

    """
    # compute person correlation and select sc according to person correlation.
    sc_mapping_dict = dict([(k,[]) for k in bulk_data.columns])
    name = bulk_data.columns[i]
    sample_cor = np.dot(bulk_data[name].values.reshape(1,bulk_data.shape[0]),sc_data.values)
    cor_index = cell_list[np.argsort(sample_cor)]
    for j, cellname in enumerate(cellname_list):
        mask = np.isin(cor_index, meta_dict[cellname])
        sub_cell = cor_index[mask]
        sub_cell = sub_cell[:int(cell_num[j])]
        sc_mapping_dict[name].extend(sub_cell)
    sample_ori = original_sc.loc[:,sc_mapping_dict[name]].sum(axis=1)
    sample = sc_data.loc[:,sc_mapping_dict[name]].sum(axis=1)
    mapped_cor = utils.pear(sample,bulk_data[name].values).item()
    print(f"sample {name} done.")

    return sample_ori.tolist(), sample.tolist(), mapped_cor, sc_mapping_dict

def bulk_mapping(frac_data,
                bulk_adata,
                sc_adata,
                n_cell=100,
                annotation_key="curated_cell_type",
                bulk_layer=None,
                sc_layer=None,
                reorder=True,
                multiprocessing=True,
                cpu_num=cpu_count()-2,
                dataset_name="",
                out_dir=".",
                normalization=True,
                filter_gene=True,
                cut_off_value=0.6,
                save=True):
    """
    Reconstruct the bulk data with the single cell data and the cell type fraction file as the reference.

    Parameters
    ----------
    frac_data: dataframe
        An :class:`~pandas.dataframe` containing the cell type feaction. Columns are cell type, rows are samples.
    bulk_adata: anndata.AnnData
        An :class:`~anndata.AnnData` containing the input bulk.
    sc_adata: anndata.AnnData
        An :class:`~anndata.AnnData` containing the single cell expression.
    n_cell: int, optional
        The number of cells contained in each sample.
    annotation_key: string, optional
        The `.obs` key where the single cell annotation is stored.: anndata.AnnData.
    bulk_layer: string, optional
        The layer to store the bulk expression.
    sc_layer: string, optional
        The layer to store the sc expression.
    reorder: boolean
        Reorder the bulk data and sc data to ensure the same gene order.
    multiprocessing: boolean, optional
        Using multiprocess or not.
    cpu_num: int, optional
        The number of cpu used in multiprocessing.
    project: string, optional
        The prefix of output file.
    out_dir: string, optional
        The path to store the output data.
    normalization: boolean, optional
        Cpm normalization on both sc and bulk data or not.
    filter_gene: boolean, optional
        Keep genes with high cosin similarity or not.
    cut_off_value: float, optional
        The cosin similarity value to filter the reconstructed genes.
    save: boolean, optional
        Save the result file or not.
        
    Returns
    -------
    Returns the preprocessed bulk data (adata) , stimualted bulk data and sc data (adata).

    """
    
    start_t = time.perf_counter()
    print("=================================================================================================")
    print('Start to mapping bulk data with single cell dataset.')
    # format data
    cell_prop = frac_data.values
    cell_matrix = np.floor(n_cell * cell_prop)
    cell_num = cell_matrix.astype(int)
    meta_data = sc_adata.obs[[annotation_key]]
    meta_dict = meta_data.groupby(meta_data[annotation_key]).groups
    for key, value in meta_dict.items():
        meta_dict[key] = np.array(value)
    cellname_list=frac_data.columns
    cell_list = np.array(sc_adata.obs.index)
    #normalization
    if normalization:
        sc_adata=utils.normalization_cpm(sc_adata,scale_factors=10000,trans_method="log")
        bulk_adata=utils.normalization_cpm(bulk_adata,scale_factors=10000,trans_method="log")
    bulk_adata.layers['mapping_ori'] = bulk_adata.X.copy()
    input_sc_data = get.count_data(sc_adata,counts_location=sc_layer)
    bulk_data = get.count_data(bulk_adata,counts_location=bulk_layer)
    
    if reorder:
        gene_order = input_sc_data.index.tolist()
        bulk_data = bulk_data.loc[gene_order]

    sc_data = utils.normal_center(input_sc_data)
    bulk_data = utils.normal_center(bulk_data)

    bulk_adata.layers['normal_center'] = bulk_data.T.values
    sample = np.zeros((cell_prop.shape[0],sc_data.shape[0]))
    mapped_cor = []
    sample_ori = np.zeros((cell_prop.shape[0],sc_data.shape[0]))
    sc_mapping_dict = dict([(k,[]) for k in bulk_data.columns])
    if multiprocessing:
        if cpu_count()<2:
            cpu_num = cpu_count()
        # compute person correlation and select sc according to person correlation.
        print(f"multiprocessing mode, cpu count is {cpu_num}")
        with Pool(cpu_num) as p:
            results = p.starmap(_bulk_mapping_parallel, [(i, cell_num[i], bulk_data, sc_data, cell_list, meta_dict, cellname_list,input_sc_data) for i in range(len(cell_num))])
        # postprocessing
        for i, (sample_ori_i,sample_i, mapped_cor_i, sc_mapping_dict_i) in enumerate(results):
            sample_ori[i]= np.array(sample_ori_i)
            sample[i] = np.array(sample_i)
            mapped_cor.append(mapped_cor_i)
            for k in sc_mapping_dict_i.keys():
                sc_mapping_dict[k].extend(sc_mapping_dict_i[k])
    else:
        for i, sample_num in tqdm(enumerate(cell_num)):
            name = bulk_data.columns[i]
            sample_cor = np.dot(bulk_data[name].values.reshape(1,bulk_data.shape[0]),sc_data.values)
            cor_index = cell_list[np.argsort(sample_cor)]
            for j, cellname in enumerate(cellname_list):
                mask = np.isin(cor_index, meta_dict[cellname])
                sub_cell = cor_index[mask]
                sub_cell = sub_cell[:int(sample_num[j])]
                sc_mapping_dict[name].extend(sub_cell)
            print(f"sample {name} done.")
            sample_ori = input_sc_data.loc[:,sc_mapping_dict[name]].sum(axis=1)
            sample = sc_data.loc[:,sc_mapping_dict[name]].sum(axis=1)
            mapped_cor_i = utils.pear(sample,bulk_data[name].values).item()
            mapped_cor.append(mapped_cor_i)
    print('initial mapping solution:',"min correlation", min(mapped_cor),"average correlation",np.mean(mapped_cor),"max correlation", max(mapped_cor))
    bulk_adata.obsm['cell_fraction'] = frac_data
    bulk_adata.obsm['cell_number']=pd.DataFrame(cell_matrix,index=frac_data.index,columns=frac_data.columns)
    bulk_adata.layers['mapping'] = sample/n_cell
    bulk_adata.layers['mapping_ori'] = sample_ori/n_cell
    if filter_gene:
        from sklearn.metrics.pairwise import cosine_similarity
        gene_list = []
        similarity_list=[]
        data_ori = pd.DataFrame(bulk_adata.X,index=bulk_adata.obs_names,columns=bulk_adata.var_names)
        data_mapping = pd.DataFrame(bulk_adata.layers['mapping_ori'],index=bulk_adata.obs_names,columns=bulk_adata.var_names)
        for gene in bulk_adata.var_names:
            similarity = cosine_similarity(data_ori[gene].values.reshape(1, -1), data_mapping[gene].values.reshape(1, -1))
            if similarity > 0.6:
                similarity_list.append(similarity[0][0])
                gene_list.append(gene)
        bulk_adata=bulk_adata[:,gene_list].copy()
        print('Gene cosin similarity:',"min value", min(similarity_list),"average value",np.mean(similarity_list),"max value", max(similarity_list))
        print(f'The number of reconstructed gene:{len(gene_list)}')  
    print(f'Time to finish mapping: {round(time.perf_counter() - start_t, 2)} seconds')
    print("=========================================================================================================================================")
    if save:
        out_dir = utils.check_paths(f'{out_dir}/filtered')
        bulk_adata.write_h5ad(f"{out_dir}/bulk_data_{dataset_name}.h5ad")

    return bulk_adata,sc_mapping_dict




def st_mapping(st_adata,sc_adata,deconv_results,solution_method="cytospace"):
    if solution_method=="cytospace":
        main_cytospace(st_adata,sc_adata,deconv_results)
    return 