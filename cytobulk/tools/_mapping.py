
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
from ._st_reconstruction import *
import random
import sys
from scipy.spatial import distance

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

def _run_st_mapping(st_adata,
                    sc_adata,
                    seed=1,
                    annotation_key='celltype_minor',
                    sc_downsample=True,
                    scRNA_max_transcripts_per_cell=1500,
                    sampling_method="duplicates"):

    np.random.seed(seed)
    random.seed(seed)
    # read data
    st_adata.var_names_make_unique()
    sc_adata.var_names_make_unique()
    st_data = get.count_data(st_adata)
    sc_data = get.count_data(sc_adata)
    cell_type_data = get.meta(sc_adata,columns=annotation_key)
    print(cell_type_data)
    coordinates_data = get.coords(st_adata)
    deconv_result = get.meta(st_adata,position_key="uns",columns="deconv")
    n_cells_per_spot_data = get.meta(st_adata,position_key="obsm",columns="cell_num")
    cell_number_to_node_assignment = n_cells_per_spot_data.astype(int)
    intersect_genes = st_data.index.intersection(sc_data.index)
    # preprocess
    scRNA_data_sampled = sc_data.loc[intersect_genes, :]
    st_data = st_data.loc[intersect_genes, :]
    # get cell num for each spot
    fraction_spot_num = deconv_result.mul(cell_number_to_node_assignment.values,axis=0)
    fraction_spot_num = fraction_spot_num.round().astype(int)
    number_of_cells = np.sum(cell_number_to_node_assignment)
    #cell_type_numbers_int = get_cell_type_fraction(number_of_cells, deconv_result)
    #print(cell_type_numbers_int)
    cell_type_numbers_int = fraction_spot_num.sum().to_frame()
    cell_type_numbers_int.columns=[0]
    print(cell_type_numbers_int)
    

    ### Sample scRNA_data
    print("Down/up sample of scRNA-seq data according to estimated cell type fractions")
    t0 = time.perf_counter()
    # downsample scRNA_data_sampled to equal transcript counts per cell
    # so that the assignment is not dependent on expression level
    if sc_downsample:
        scRNA_data_sampled = downsample(scRNA_data_sampled, scRNA_max_transcripts_per_cell)

    # sample scRNA_data based on cell type composition
    # cell count in scRNA_data_sampled will be equal to cell count (not spot count) in ST data
    # additionally, the cells in scRNA_data_sampled will be in the order of cell types in cell_type_numbers_int
    scRNA_data_sampled =\
        sample_single_cells(scRNA_data_sampled, cell_type_data, cell_type_numbers_int, sampling_method, seed)

    print(f"Time to down/up sample scRNA-seq data: {round(time.perf_counter() - t0, 2)} seconds")

    # normalize data; output is an np.ndarray
    scRNA_norm_np = utils.normalize_data(scRNA_data_sampled.to_numpy())
    st_norm_np = utils.normalize_data(st_data.to_numpy())

    # regenerate pandas dataframe from the normalized data
    scRNA_norm_data = pd.DataFrame(scRNA_norm_np, index=scRNA_data_sampled.index, columns=scRNA_data_sampled.columns)
    st_norm_data = pd.DataFrame(st_norm_np, index=st_data.index, columns=st_data.columns)




    ''' 
    index_sc_list = partition_indices(np.arange(scRNA_data_sampled.shape[1]), shuffle=False)

    subsampled_cell_number_to_node_assignment_list=None
    index_st_list=None
    distance_metric="Euclidean"
    if (index_st_list is not None) and (subsampled_cell_number_to_node_assignment_list is not None):
        raise ValueError("index_st_list and subsampled_cell_number_to_node_assignment_list cannot both be specified")

    # normalize data; output is an np.ndarray
    scRNA_norm_np = normalize_data(scRNA_data_sampled.to_numpy())
    st_norm_np = normalize_data(st_data.to_numpy())

    # regenerate pandas dataframe from the normalized data
    scRNA_norm_data = pd.DataFrame(scRNA_norm_np, index=scRNA_data_sampled.index, columns=scRNA_data_sampled.columns)
    st_norm_data = pd.DataFrame(st_norm_np, index=st_data.index, columns=st_data.columns)

    expressions_tpm_scRNA_log = scRNA_norm_data.iloc[:, index_sc_list[0]].to_numpy()
    expressions_tpm_st_log = st_norm_data.to_numpy()
    '''  
    # cell types are not specified; estimating from cell type fraction data
    assigned_locations_list = []
    cell_ids_selected_list=[]
    for cells in cell_type_numbers_int.index.tolist():
        scRNA_data_sampled_cell_type = scRNA_data_sampled.columns.tolist()
        cell_type_index = cell_type_data[cell_type_data.values == cells].index.tolist()
        cell_type_selected_index=np.intersect1d(scRNA_data_sampled_cell_type,cell_type_index).tolist()
        #scRNA_data_sampled_cell_type = scRNA_data_sampled.loc[:,cell_type_selected_index]
        #index_sc_list = partition_indices(np.arange(scRNA_data_sampled_cell_type.shape[1]), shuffle=False)
        st_selected_index = fraction_spot_num[fraction_spot_num[cells]>0].index.to_list()
        cell_number_to_node_assignment_cell_type = fraction_spot_num.loc[st_selected_index,cells]
        expressions_tpm_scRNA_log = scRNA_norm_data.loc[:,cell_type_selected_index]
        expressions_tpm_st_log = st_norm_data.loc[:,st_selected_index]
        sub_coordinates_data  = coordinates_data.loc[st_selected_index,:]
        not_assigned_pos = np.arange(np.sum(cell_number_to_node_assignment_cell_type))
        print("st")
        print(len(not_assigned_pos))
        not_assigned_cell = np.arange(expressions_tpm_scRNA_log.shape[1])
        print("sc")
        print(len(not_assigned_cell))
        lap_expressions_tpm_st_log = expressions_tpm_st_log
        lap_expressions_tpm_scRNA_log = expressions_tpm_scRNA_log
        #total_cell_names = np.array(cell_type_selected_index)
        #total_spot_names = np.array(st_selected_index)
        total_cost = 0
        while (len(not_assigned_pos)):
            cost_matrix,lap_expressions_tpm_st_log = build_cost_matrix(lap_expressions_tpm_st_log,lap_expressions_tpm_scRNA_log, 0.2,cell_number_to_node_assignment_cell_type)
            #res_matrix = np.zeros(lap_expressions_tpm_st_log.shape[1])
            assignment_mat, cost = lap(cost_matrix)
            assignment_mat = assignment_mat.astype(int)
            assigned_pos_index = np.where(assignment_mat != -1)[0]
            assigned_cell_index = assignment_mat[assigned_pos_index]
            assigned_locations_list += list(lap_expressions_tpm_st_log.columns[assigned_pos_index])
            cell_ids_selected_list += list(lap_expressions_tpm_scRNA_log.columns[assigned_cell_index])
            #res_matrix[not_assigned_pos[assigned_pos_index]] = not_assigned_cell[assigned_cell_index]
            not_assigned_pos_index = np.where(assignment_mat == -1)[0]
            not_assigned_pos = not_assigned_pos[not_assigned_pos_index]
            
            #total_spot_names = total_spot_names[not_assigned_pos_index]
            not_assigned_cell_index = np.isin(np.arange(len(not_assigned_cell)), assigned_cell_index, invert=True)
            #total_cell_names = total_cell_names[not_assigned_cell_index]
            not_assigned_cell = not_assigned_cell[not_assigned_cell_index]
            not_assigned_num = len(not_assigned_pos_index)
            print("assignment left: ", not_assigned_num)
            total_cost += cost
            lap_expressions_tpm_st_log = lap_expressions_tpm_st_log.iloc[:, not_assigned_pos]
            lap_expressions_tpm_scRNA_log= lap_expressions_tpm_scRNA_log.iloc[:, not_assigned_cell]
            not_assigned_cell = np.arange(lap_expressions_tpm_scRNA_log.shape[1])
            cell_number_to_node_assignment_cell_type = np.array([1]*len(not_assigned_pos))
            not_assigned_pos = np.arange(np.sum(cell_number_to_node_assignment_cell_type))
            
        #print(cost_matrix)
        #location_repeat = np.zeros(expressions_tpm_st_log.shape[1])
        ##location_repeat = np.repeat(np.arange(len(cell_number_to_node_assignment_cell_type)), cell_number_to_node_assignment_cell_type)
        #location_repeat = location_repeat.astype(int)
        #print(expressions_tpm_st_log.columns[location_repeat])
        #expressions_tpm_st_log = expressions_tpm_st_log.loc[:, expressions_tpm_st_log.columns[location_repeat]]
        #print(expressions_tpm_st_log)
    reconstructed_sc = pd.DataFrame({"spot_id":assigned_locations_list,"cell_id":cell_ids_selected_list})
    print(reconstructed_sc)
        #while (len(not_assigned_pos)):

    '''
        if (len(st_selected_index) > 0) and (len(cell_type_selected_index) > 0):
            assigned_locations, cell_ids_selected =\
                apply_linear_assignment(scRNA_data_sampled_cell_type, st_data_sub, sub_coordinates_data, cell_number_to_node_assignment_cell_type,
                                        solver_method, solver, seed, distance_metric, number_of_processors,index_sc_list)
            assigned_locations_list.append(assigned_locations)
            cell_ids_selected_list.append(cell_ids_selected)


    for cells in cell_type_numbers_int.index.tolist():
        scRNA_data_sampled_cell_type = scRNA_data_sampled.columns.tolist()
        cell_type_index = cell_type_data[cell_type_data[annotation_key] == cells].index.tolist()
        cell_type_selected_index=np.intersect1d(scRNA_data_sampled_cell_type,cell_type_index).tolist()
        scRNA_data_sampled_cell_type = scRNA_data_sampled.loc[:,cell_type_selected_index]
        index_sc_list = partition_indices(np.arange(scRNA_data_sampled_cell_type.shape[1]), shuffle=False)
        st_selected_index = fraction_spot_num[fraction_spot_num[cells]>0].index.to_list()
        cell_number_to_node_assignment_cell_type = fraction_spot_num.loc[st_selected_index,cells]
        st_data_sub=st_data.loc[:,st_selected_index]
        sub_coordinates_data  = coordinates_data.loc[st_selected_index,:]
        if (len(st_selected_index) > 0) and (len(cell_type_selected_index) > 0):
            assigned_locations, cell_ids_selected =\
                apply_linear_assignment(scRNA_data_sampled_cell_type, st_data_sub, sub_coordinates_data, cell_number_to_node_assignment_cell_type,
                                        solver_method, solver, seed, distance_metric, number_of_processors,index_sc_list)
            assigned_locations_list.append(assigned_locations)
            cell_ids_selected_list.append(cell_ids_selected)
    assigned_locations = pd.concat(assigned_locations_list)
    cell_ids_selected = np.concatenate(cell_ids_selected_list, axis=0)

    
                
    
       
    print(f"Total time to run CytoSPACE core algorithm: {round(time.perf_counter() - t0_core, 2)} seconds")
    with open(fout_log,"a") as f:
        f.write(f"Time to run CytoSPACE core algorithm: {round(time.perf_counter() - t0_core, 2)} seconds\n")

    ### Save results
    print('Saving results ...')
    
    # identify unmapped spots
    mapped_spots = assigned_locations.index
    unmapped_spots = np.setdiff1d(list(all_spot_ids), list(mapped_spots)).tolist()

    if len(unmapped_spots) > 0:
        unassigned_locations  = coordinates_data.loc[unmapped_spots]
        unassigned_locations.index = unassigned_locations.index.str.replace("SPOT_", "")
        unassigned_locations["Number of cells"] = 0
        unassigned_locations.to_csv(f"{output_path}/{output_prefix}unassigned_locations.csv", index=True)
        print(f"{len(unmapped_spots)} spots had no cells mapped to them. Saved unfiltered version of assigned locations to {output_path}/{output_prefix}unassigned_locations.csv")

    
    
    save_results(output_path, output_prefix, cell_ids_selected, scRNA_data_sampled if sampling_method == "place_holders" else scRNA_data,
                 assigned_locations, cell_type_data, sampling_method, single_cell)

    if not plot_off:
        if single_cell:
            plot_results(output_path, output_prefix, max_num_cells=max_num_cells_plot, single_cell_ST_mode=True)
        else:
            plot_results(output_path, output_prefix, coordinates_data=coordinates_data, geometry=geometry, num_cols=num_column, max_num_cells=max_num_cells_plot)

    print(f"Total execution time: {round(time.perf_counter() - start_time, 2)} seconds")
    with open(fout_log,"a") as f:
        f.write(f"Total execution time: {round(time.perf_counter() - start_time, 2)} seconds")
    '''


def st_mapping(st_adata,sc_adata):
    start_t = time.perf_counter()
    print("=================================================================================================")
    print('Start to mapping bulk data with single cell dataset.')
    _run_st_mapping(st_adata,sc_adata)
    print(f'Time to finish mapping: {round(time.perf_counter() - start_t, 2)} seconds')
    print("=========================================================================================================================================")
    
    return 