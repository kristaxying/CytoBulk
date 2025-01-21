
import pandas as pd
import numpy as np
from tqdm import tqdm
from .. import get
from .. import utils
import time
import ot
from pkg_resources import resource_filename
import sys
import scanpy as sc
from multiprocessing import Pool, cpu_count
import numpy as np
from ._st_reconstruction import *
from ._he_inference import *
import random

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

def bulk_mapping(bulk_adata,
                sc_adata,
                n_cell=100,
                annotation_key="curated_cell_type",
                bulk_layer=None,
                sc_layer=None,
                reorder=True,
                multiprocessing=True,
                cpu_num=cpu_count()-4,
                project="",
                out_dir=".",
                normalization=True,
                filter_gene=True,
                save=True):
    """
    Reconstruct bulk data using single-cell data and cell type fractions.

    This function maps bulk expression data to single-cell expression data using
    cell type fraction information and various preprocessing steps.

    Parameters
    ----------
    bulk_adata : anndata.AnnData
        An :class:`~anndata.AnnData` object containing the input bulk data.

    sc_adata : anndata.AnnData
        An :class:`~anndata.AnnData` object containing the single-cell expression data.

    n_cell : int, optional
        Number of cells per bulk sample.

    annotation_key : string, optional
        Key in `sc_adata.obs` for single-cell annotations.

    bulk_layer : string, optional
        Layer in `bulk_adata` to use for bulk expression data.

    sc_layer : string, optional
        Layer in `sc_adata` to use for single-cell expression data.

    reorder : bool, optional (default: True)
        Whether to reorder genes to ensure consistency between bulk and single-cell data.

    multiprocessing : bool, optional (default: True)
        Whether to use multiprocessing for efficiency.

    cpu_num : int, optional
        Number of CPUs to use if multiprocessing is enabled.

    project : string, optional
        Prefix for output files.

    out_dir : string, optional
        Directory to store output files.

    normalization : bool, optional (default: True)
        Whether to apply CPM normalization to data.

    filter_gene : bool, optional (default: True)
        Whether to filter genes based on cosine similarity.

    cut_off_value : float, optional (default: 0.6)
        Threshold for cosine similarity when filtering genes.

    save : bool, optional (default: True)
        Whether to save the result files.

    Returns
    -------
    bulk_adata : anndata.AnnData
        The processed bulk data with mapping results.

    df : pandas.DataFrame
        DataFrame containing the mapping of bulk samples to single-cell IDs.
    """
    
    start_t = time.perf_counter()
    print("=================================================================================================")
    print('Start to mapping bulk data with single cell dataset.')
    # format data
    bulk_adata.var_names_make_unique()
    sc_adata.var_names_make_unique()
    intersect_gene = bulk_adata.var_names.intersection(sc_adata.var_names)
    bulk_adata = bulk_adata[:,intersect_gene]
    sc_adata = sc_adata[:,intersect_gene]
    cell_prop = bulk_adata.uns['deconv']
    cell_matrix = np.floor(n_cell * cell_prop)
    cell_num = cell_matrix.astype(int)
    meta_data = sc_adata.obs[[annotation_key]]
    meta_dict = meta_data.groupby(meta_data[annotation_key]).groups
    for key, value in meta_dict.items():
        meta_dict[key] = np.array(value)
    cellname_list=cell_prop.columns
    cell_list = np.array(sc_adata.obs_names)
    #normalization
    bulk_adata.layers['mapping_ori'] = bulk_adata.X.copy()
    if normalization:
        sc_adata=utils.normalization_cpm(sc_adata,scale_factors=100000,trans_method="log")
        bulk_adata=utils.normalization_cpm(bulk_adata,scale_factors=100000,trans_method="log")
    bulk_adata.layers['mapping_nor'] = bulk_adata.X.copy()
    input_sc_data = get.count_data(sc_adata,counts_location=sc_layer)
    bulk_data = get.count_data(bulk_adata,counts_location=bulk_layer)

    if reorder:
        intersect_gene = input_sc_data.index.intersection(bulk_data.index)
        input_sc_data = input_sc_data.loc[intersect_gene,:]
        bulk_data = bulk_data.loc[intersect_gene,:]

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
        with Pool(int(cpu_num)) as p:
            results = p.starmap(_bulk_mapping_parallel, [(i, cell_num.iloc[i,:], bulk_data, sc_data, cell_list, meta_dict, cellname_list, input_sc_data)
             for i in range(cell_num.shape[0])])
            print(results)
        # postprocessing
        for i, (sample_ori_i,sample_i, mapped_cor_i, sc_mapping_dict_i) in enumerate(results):
            sample_ori[i]= np.array(sample_ori_i)
            sample[i] = np.array(sample_i)
            mapped_cor.append(mapped_cor_i)
            for k in sc_mapping_dict_i.keys():
                sc_mapping_dict[k].extend(sc_mapping_dict_i[k])
    else:
        for index_num, (i, sample_num) in enumerate(tqdm(cell_num.iterrows())):
            sample_cor = np.dot(bulk_data[i].values.reshape(1,bulk_data.shape[0]),sc_data.values)
            cor_index = cell_list[np.argsort(sample_cor)]
            for j, cellname in enumerate(cellname_list):
                mask = np.isin(cor_index, meta_dict[cellname])
                sub_cell = cor_index[mask]
                sub_cell = sub_cell[:int(sample_num[j])]
                sc_mapping_dict[i].extend(sub_cell)
            print(f"sample {i} done.")
            sample_ori[index_num,:] = input_sc_data.loc[:,sc_mapping_dict[i]].sum(axis=1)
            sample[index_num,:] = sc_data.loc[:,sc_mapping_dict[i]].sum(axis=1)
            mapped_cor_i = utils.pear(sample[index_num,:],bulk_data[i].values).item()
            mapped_cor.append(mapped_cor_i)
    print('initial mapping solution:',"min correlation", min(mapped_cor),"average correlation",np.mean(mapped_cor),"max correlation", max(mapped_cor))
    bulk_adata.obsm['cell_number']=pd.DataFrame(cell_matrix,index=cell_prop.index,columns=cell_prop.columns)
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
        out_dir = utils.check_paths(f'{out_dir}/output')
        df = pd.DataFrame([(k, v) for k, lst in sc_mapping_dict.items() for v in lst], columns=['sample_id', 'cell_id'])
        df.to_csv(f"{out_dir}/bulk_data_{project}_mapping.csv")
        bulk_adata.write_h5ad(f"{out_dir}/bulk_data_{project}_mapping.h5ad")

    return df,bulk_adata


def _run_st_mapping(st_adata,
                    sc_adata,
                    seed=0,
                    annotation_key='celltype_minor',
                    sc_downsample=False,
                    scRNA_max_transcripts_per_cell=1500,
                    out_dir='.',
                    project='test',
                    mean_cell_numbers=8,
                    save_reconstructed_st=True):
    np.random.seed(seed)
    random.seed(seed)
    # read data
    st_adata.var_names_make_unique()
    sc_adata.var_names_make_unique()
    st_data = get.count_data(st_adata)
    sc_data = get.count_data(sc_adata)
    cell_type_data = get.meta(sc_adata,columns=annotation_key)
    coordinates_data = get.coords(st_adata)
    deconv_result = get.meta(st_adata,position_key="uns",columns="deconv")
    intersect_spot = st_data.columns.intersection(deconv_result.index)
    st_data = st_data.loc[:,intersect_spot]
    if "cell_num" in st_adata.obsm:
        n_cells_per_spot_data = get.meta(st_adata,position_key="obsm",columns="cell_num")
    else:
        n_cells_per_spot_data = estimate_cell_num(st_data, mean_cell_numbers)
    #n_cells_per_spot_data.to_csv(f"{out_dir}/output/n_cells_per_spot_data.csv")
    cell_number_to_node_assignment = n_cells_per_spot_data.astype(int)
    intersect_genes = st_data.index.intersection(sc_data.index)
    
    # preprocess
    scRNA_data_sampled = sc_data.loc[intersect_genes, :]
    st_data = st_data.loc[intersect_genes, :]
    fraction_spot_num = deconv_result.mul(cell_number_to_node_assignment.values,axis=0)
    #fraction_spot_num.to_csv(f"{out_dir}/output/fraction_spot_num.csv")
    fraction_spot_num = fraction_spot_num.apply(modify_row, axis=1)
    #fraction_spot_num.to_csv(f"{out_dir}/output/fraction_spot_num_motified.csv")
    cell_type_numbers_int = fraction_spot_num.sum().to_frame()
    #cell_type_numbers_int.to_csv(f"{out_dir}/output/cell_type_numbers_int.csv")
    cell_type_numbers_int.columns=[0]

    

    ### Sample scRNA_data
    print("Down/up sample of scRNA-seq data according to estimated cell type fractions")
    t0 = time.perf_counter()

    if sc_downsample:
        scRNA_data_sampled = downsample(scRNA_data_sampled, scRNA_max_transcripts_per_cell)

    print(f"Time to down/up sample scRNA-seq data: {round(time.perf_counter() - t0, 2)} seconds")

    # normalize data; output is an np.ndarray
    scRNA_norm_np = utils.normalize_data(scRNA_data_sampled.to_numpy())
    st_norm_np = utils.normalize_data(st_data.to_numpy())

    # regenerate pandas dataframe from the normalized data
    scRNA_norm_data = pd.DataFrame(scRNA_norm_np, index=scRNA_data_sampled.index, columns=scRNA_data_sampled.columns)

    st_norm_data = pd.DataFrame(st_norm_np, index=st_data.index, columns=st_data.columns)


    # cell types are not specified; estimating from cell type fraction data
    assigned_locations_list = []
    cell_ids_selected_list=[]
    #round_item=0
    for cells in cell_type_numbers_int.index.tolist():
        scRNA_data_sampled_cell_type = scRNA_data_sampled.columns.tolist()
        cell_type_index = cell_type_data[cell_type_data.values == cells].index.tolist()
        cell_type_selected_index=np.intersect1d(scRNA_data_sampled_cell_type,cell_type_index).tolist()
        st_selected_index = fraction_spot_num[fraction_spot_num[cells]>0].index.to_list()
        cell_number_to_node_assignment_cell_type = fraction_spot_num.loc[st_selected_index,cells]
        expressions_tpm_scRNA_log = scRNA_norm_data.loc[:,cell_type_selected_index]
        expressions_tpm_st_log = st_norm_data.loc[:,st_selected_index]
        #sub_coordinates_data  = coordinates_data.loc[st_selected_index,:]
        not_assigned_pos = np.arange(np.sum(cell_number_to_node_assignment_cell_type))
        not_assigned_cell = np.arange(expressions_tpm_scRNA_log.shape[1])
        lap_expressions_tpm_st_log = expressions_tpm_st_log
        lap_expressions_tpm_scRNA_log = expressions_tpm_scRNA_log
        total_cost = 0
        back_up_sc = lap_expressions_tpm_scRNA_log
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
            #print("assignment left: ", not_assigned_num)
            total_cost += cost
            lap_expressions_tpm_st_log = lap_expressions_tpm_st_log.iloc[:, not_assigned_pos]
            if (len(not_assigned_cell)==0)&(len(not_assigned_pos)>0):
                lap_expressions_tpm_scRNA_log = back_up_sc
            else:
                lap_expressions_tpm_scRNA_log = lap_expressions_tpm_scRNA_log.iloc[:, not_assigned_cell]
            not_assigned_cell = np.arange(lap_expressions_tpm_scRNA_log.shape[1])
            cell_number_to_node_assignment_cell_type = np.array([1]*len(not_assigned_pos))
            not_assigned_pos = np.arange(np.sum(cell_number_to_node_assignment_cell_type))
            
    reconstructed_sc = pd.DataFrame({"spot_id":assigned_locations_list,"cell_id":cell_ids_selected_list})
    reconstructed_sc.to_csv(f"{out_dir}/output/{project}_reconstructed_st.csv")
    if save_reconstructed_st:
        spot_expression = {}
        for spot_id in np.unique(reconstructed_sc['spot_id']):
            cells_in_spot = reconstructed_sc[reconstructed_sc['spot_id'] == spot_id]['cell_id']
            expression_sum = scRNA_data_sampled[cells_in_spot].sum(axis=1)
            spot_expression[spot_id] = expression_sum
        new_adata = sc.AnnData(X=pd.DataFrame(spot_expression).T.values, var=pd.DataFrame(index=scRNA_data_sampled.index))
        new_adata.obs_names = list(spot_expression.keys())
        new_adata.var_names = scRNA_norm_data.index
        st_data=st_data[new_adata.obs_names]
        new_adata.layers['original_st'] = st_data.values.T
        new_adata.obsm['spatial'] = coordinates_data.loc[new_adata.obs_names,:].values
        new_adata.uns = st_adata.uns
        new_adata.write(f"{out_dir}/output/reconstructed_{project}_st.h5ad")

    return reconstructed_sc,new_adata

  

def st_mapping(st_adata,
               sc_adata,
               out_dir,
               project,
               annotation_key,
               **kwargs):
    """
    Run spatial transcriptomics mapping with single-cell RNA-seq data.

    This function maps spatial transcriptomics (ST) data to single-cell RNA-seq (scRNA-seq) data. It aligns cell type compositions and estimates spatial distributions.

    Parameters
    ----------
    st_adata : anndata.AnnData
        An :class:`~anndata.AnnData` object containing spatial transcriptomics data.
    
    sc_adata : anndata.AnnData
        An :class:`~anndata.AnnData` object containing single-cell RNA-seq data.
    
    seed : int, optional (default: 0)
        Seed for random number generation to ensure reproducibility.
    
    annotation_key : string, optional (default: 'celltype_minor')
        Key in `sc_adata` for cell type annotations.
    
    sc_downsample : bool, optional (default: False)
        Whether to downsample scRNA-seq data to a maximum number of transcripts per cell.
    
    scRNA_max_transcripts_per_cell : int, optional (default: 1500)
        Maximum number of transcripts per cell for downsampling.
    
    sampling_method : string, optional (default: 'duplicates')
        Method for sampling single cells based on cell type composition.
    
    out_dir : string, optional (default: '.')
        Directory to save output files.
    
    project : string, optional (default: 'test')
        Project name for output file naming.
    
    mean_cell_numbers : int, optional (default: 8)
        Average number of cells per spot used for estimation.
    
    save_reconstructed_st : bool, optional (default: True)
        Whether to save the reconstructed spatial transcriptomics data.

    Returns
    -------
    reconstructed_sc : pandas.DataFrame
        DataFrame containing the mapping of single-cell IDs to spatial spot IDs.
    """
    start_t = time.perf_counter()
    print("=================================================================================================")
    print('Start to mapping bulk data with single cell dataset.')
    reconstructed_sc=_run_st_mapping(st_adata = st_adata,
                                     sc_adata = sc_adata,
                                     out_dir = out_dir,
                                     project = project,
                                     annotation_key=annotation_key,
                                     **kwargs)
    print(f'Time to finish mapping: {round(time.perf_counter() - start_t, 2)} seconds')
    print("=========================================================================================================================================")
    
    return reconstructed_sc


def he_mapping(image_dir,
               out_dir,
               project,
               lr_data = None,
               sc_adata = None,
               annotation_key="curated_celltype",
               k_neighbor=30,
               alpha=0.5,
               mapping_sc=True,
               **kwargs):

    """
    Run H&E-stained image cell type mapping with single-cell RNA-seq data.

    This function predicts cell types from H&E-stained histology images and aligns them with single-cell RNA-seq (scRNA-seq) data using optimal transport. It computes spatial distributions and matches cell types between the image and single-cell data.

    Parameters
    ----------
    image_dir : str
        Path to the directory containing H&E-stained images.

    out_dir : str
        Directory where the output files will be saved.

    project : str
        Name of the project, used for naming output files.

    lr_data : pandas.DataFrame, optional (default: None)
        A DataFrame containing ligand-receptor pair data with columns 'ligand' and 'receptor'.

    sc_adata : anndata.AnnData, optional (default: None)
        An :class:`~anndata.AnnData` object containing single-cell RNA-seq data with gene expression profiles.

    annotation_key : str, optional (default: "curated_celltype")
        Key in `sc_adata.obs` for cell type annotations.

    k_neighbor : int, optional (default: 30)
        Number of neighbors to consider when constructing the graph for H&E image data.

    alpha : float, optional (default: 0.5)
        Trade-off parameter for the Fused Gromov-Wasserstein optimal transport, controlling the balance between graph structure and feature matching (value between 0 and 1).

    mapping_sc : bool, optional (default: True)
        Whether to perform mapping between H&E image cell data and single-cell RNA-seq data. If False, only H&E image cell type predictions are returned.

    **kwargs : dict
        Additional arguments (not used in this implementation).

    Returns
    -------
    cell_coordinates : pandas.DataFrame
        DataFrame containing cell coordinates and their predicted cell types from H&E-stained images.

    df : pandas.DataFrame
        DataFrame containing matching results between H&E image cells and single-cell data, including spatial coordinates, cell types, and matched single-cell IDs.

    filtered_adata : anndata.AnnData
        A filtered :class:`~anndata.AnnData` object containing only the single-cell data that matches with cells from H&E-stained images.
    """
    start_t = time.perf_counter()
    file_dir = resource_filename(__name__, 'model/pretrained_models/')
    file_name = 'DeepCMorph_Datasets_Combined_41_classes_acc_8159.pth'

    # The download URL for the file (replace with the actual URL)
    download_url = "https://drive.google.com/file/d/1M5dNB5XtrboQcxuI59rPNdR29nLe6UFC/view?usp=sharing"

    # Ensure the file exists; if not, download it
    get.ensure_file_exists(file_dir, file_name, download_url)
    cell_coordinates = inference_cell_type_from_he_image(image_dir,
                                                         out_dir,
                                                         project)
    if mapping_sc:
        print("preprocessing of single cell data")
        lr_genes = np.unique(np.concatenate((lr_data['ligand'].values, lr_data['receptor'].values)))
        sc.pp.filter_cells(sc_adata, min_genes=200)  # filter
        sc.pp.filter_genes(sc_adata, min_cells=3)   # filter
        sc.pp.normalize_total(sc_adata, target_sum=1e4)  # nor
        sc.pp.log1p(sc_adata)  # log
        common_gene = np.intersect1d(lr_genes, sc_adata.var_names)
        sc_adata = sc_adata[:, common_gene].copy()
        lr_data = lr_data[
            (lr_data['ligand'].isin(common_gene)) & (lr_data['receptor'].isin(common_gene))
        ].copy()

        adata_cell_types = set(sc_adata.obs[annotation_key].unique())
        coordinates_cell_types = set(cell_coordinates["cell_type"].unique())


        # common_celltype
        common_cell_types = adata_cell_types.intersection(coordinates_cell_types)
        print(f"Common cell types: {common_cell_types}")

        # filter adata and cell_coordinates
        sc_adata = sc_adata[sc_adata.obs[annotation_key].isin(common_cell_types), :].copy()
        cell_coordinates = cell_coordinates[cell_coordinates["cell_type"].isin(common_cell_types)].copy()


        print("loading graph for H&E image...")
        graph1_adj, graph1_labels = load_graph1(cell_coordinates,k=k_neighbor)

        print("loading graph for single cell data with LR affinity...")
        graph2_dist, graph2_labels,sc_adata = load_graph2_with_LR_affinity(sc_adata, 
                                                                graph1_labels,
                                                                lr_data,
                                                                annotation_key)

        graph2_dist = np.nan_to_num(graph2_dist, nan=np.nanmax(graph2_dist), posinf=np.nanmax(graph2_dist), neginf=0)

        print("compute cost matrix")
        cost_matrix = construct_cost_matrix(graph1_labels, graph2_labels)

        cost_matrix = np.nan_to_num(cost_matrix, nan=np.nanmax(cost_matrix), posinf=np.nanmax(cost_matrix), neginf=0)

        print("optimal transport...")
        p = np.ones(graph1_adj.shape[0]) / graph1_adj.shape[0]
        q = np.ones(graph2_dist.shape[0]) / graph2_dist.shape[0]

        p = np.nan_to_num(p, nan=1.0 / len(p), posinf=1.0 / len(p), neginf=0)
        q = np.nan_to_num(q, nan=1.0 / len(q), posinf=1.0 / len(q), neginf=0)

        # Fused Gromov-Wasserstein
        '''
        gw_trans, log = ot.gromov.BAPG_fused_gromov_wasserstein(
            cost_matrix, graph1_adj, graph2_dist, p, q, alpha=alpha, log=True
        )
        '''
        ot_plan = ot.gromov.fused_gromov_wasserstein(
        cost_matrix, graph1_adj, graph2_dist, p, q, alpha=alpha, loss_fun='square_loss'
        )
        print(f'Time to finish mapping: {round(time.perf_counter() - start_t, 2)} seconds')
        print("=========================================================================================================================================")
        # Step 5: matching file
        print("build matching file...")
        locations = list(range(graph1_adj.shape[0])) 
        cells = list(range(graph2_dist.shape[0])) 
        matches = extract_matching_relationships(ot_plan, locations, cells)
        df = pd.DataFrame(matches, columns=["location", "cell"])
        location_mapping = cell_coordinates[["x", "y"]].to_dict(orient="index")
        df["x"] = df["location"].map(lambda c: cell_coordinates['x'].values[c])
        df["y"] = df["location"].map(lambda c: cell_coordinates['y'].values[c])
        df["he_cell_type"] = df["location"].map(lambda c: cell_coordinates['cell_type'].values[c])
        df["cell_type"] = df["cell"].map(lambda c: graph2_labels[c])
        df["cell_id"] = df["cell"].map(lambda c: sc_adata.obs_names[c])
        df.to_csv(f"{out_dir}/{project}_matching_results.csv", index=False)
        df_cell_ids = df['cell_id']
        adata_cell_ids = sc_adata.obs_names 
        common_cell_ids = set(df_cell_ids).intersection(set(adata_cell_ids))
        filtered_df = df[df['cell_id'].isin(common_cell_ids)].copy()
        filtered_adata = sc_adata[sc_adata.obs_names.isin(common_cell_ids)].copy()
        filtered_df = filtered_df.set_index('cell_id')
        #filtered_adata.obs = pd.DataFrame()
        filtered_adata.obs = filtered_df
        filtered_adata.var.index.name = "gene"
        filtered_adata.write_h5ad(f"{out_dir}/{project}_matching_adata.h5ad")
        return cell_coordinates,df,filtered_adata
    else:
        return cell_coordinates

    
    