import numpy as np
import pandas as pd
from .. import utils
from .. import get
from ._rpackage import find_marker_giotto
from ._filtering import qc_bulk_sc,qc_st_sc,high_variable_gene
from sklearn.metrics.pairwise import cosine_similarity
from os.path import exists
import json
import time




def _check_cell_type(sc_adata,cell_type_key="cell_type",cell_list=None):
    
    """
    check the cell type in single cell whether could be found in marker gene profile.

    Parameters
    ----------
    marker_gene : dataframe
        An :class:`~pandas.dataframe` marker gene in of each cell type.
        Each column is marker genes of one cell type. the first row should be name of cell types.
    sc_adata : ~anndata.AnnData
        An :class:`~anndata.AnnData` containing the sc rna expression to filter.
    cell_type_key : string.
        The `.obs` key where the annotation is stored in sc adata.

    Returns
    -------
    Returns the filtered marker gene list.
    
    """
    sc_cells = np.unique(sc_adata.obs[cell_type_key])
    if cell_list is not None:
        common_cell = np.intersect1d(np.array(cell_list), sc_cells, assume_unique=False, return_indices=False)
    else:
        common_cell = sc_cells
    '''
    if marker_data is not None and not isinstance(marker_data, pd.DataFrame):
        marker_cells = np.intersect1d(marker_data.columns, common_cell, assume_unique=False, return_indices=False)
    else:
        marker_cells = common_cell

    
    if len(marker_cells) != len(common_cell): # filter happened
        warnings.warn("In marker gene profile, could not find all the cell type of sc adata. ")
    
    if marker_data is not None:
        marker_gene = marker_data[marker_cells].to_dict('list')
    else:
        marker_gene = dict.fromkeys(common_cell)
    '''
    return common_cell




def _join_marker(bulk_adata,sc_adata,annotation_key,marker_list,out_dir='./',dataset_name='',python_path=None):

    """
    join the auto find marker and database marker together.

    Parameters
    ----------
    raw_sc_adata: anndata.AnnData
        An :class:`~anndata.AnnData` containing the raw expression.
    annotation_key: string
        The `.obs` key where the annotation is stored.: anndata.AnnData
    marker_dict:
        The marker gene dictionary from database.

    Returns
    -------
    Returns the marker gene dictionary from both database and auto-seeking.

    """
    
    if not exists(f'{out_dir}/{dataset_name}_marker.txt'):
        print("-------------------------Start finding cell type marker genes using Gitto--------------------------")
        start_t = time.perf_counter()
        marker_gene = find_marker_giotto(sc_adata,annotation_key,out_dir,dataset_name,python_path)
        print("-------------------------------Finish marker gene detection--------------------------------------")
        print(f'Time to finish marker gene detection: {round(time.perf_counter() - start_t, 2)} seconds')
    else:
        print(f'{out_dir}/{dataset_name}_marker.txt already exists, skipping find marker.')
    
    marker_gene = pd.read_csv(f"{out_dir}/{dataset_name}_marker.txt",sep='\t')
    common_gene = sc_adata.var_names.intersection(bulk_adata.var_names)

    
    
    return utils.marker_integration(common_gene,marker_gene,marker_list)



def _normalization_data(bulk_adata,sc_adata,scale_factors=None,trans_method='log',save=False,project='',out_dir='./'):
    """
    Normalization on bulk and sc adata.
        CPM = readsMappedToGene * 1/totalNumReads * 10^4
        totalNumReads       - total number of mapped reads of a sample
        readsMappedToGene   - number of reads mapped to a selected gene

    Parameters
    ----------
    bulk_data: dataframe
        An :class:`~anndata.AnnData` containing the expression to normalization.
    sc_adata: anndata.AnnData
        An :class:`~anndata.AnnData` containing the expression to normalization.
    scale_factors:
        The number of counts to normalize every observation to before computing profiles. If `None`, no normalization is performed. 
    trans_method:
        What transformation to apply to the expression before computing the profiles. 
        - `log`: log(x+1)
        - `None`: no transformation
        

    Returns
    -------
    Returns the normalizated bulk data (adata) and sc data (adata).

    """

    print("Start normalization")
    bulk_data = utils.normalization_cpm(bulk_adata,scale_factors=scale_factors,trans_method=trans_method)
    print("Finsh bulk normalization")
    sc_data = utils.normalization_cpm(sc_adata,scale_factors=scale_factors,trans_method=trans_method)
    print("Finsh sc normalization")
    
    if save:
        new_dir = utils.check_paths(out_dir+'/filtered')
        b_data = get.count_data(bulk_data)
        c_data = get.count_data(sc_data)
        b_data.to_csv(new_dir+f"/{project}_nor_bulk.txt",sep='\t')
        c_data.to_csv(new_dir+f"/{project}_nor_sc.txt",sep='\t')

    return sc_data,bulk_data

def _filter_cells(sc_adata, bulk_adata,num=10):
    cell_name = sc_adata.obs_names
    bulk_matrix = bulk_adata.X
    sc_matrix = sc_adata.X
    similarity_matrix = cosine_similarity(bulk_matrix,sc_matrix)
    top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, :num]
    selected_indices = np.unique(top_k_indices.flatten())
    #if 10*bulk_adata.shape[0]<500:
    target_num=500
    #else:
        #target_num=10*bulk_adata.shape[0]
    while len(selected_indices)<target_num:
          num=int(num*1.25)
          top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, :num]
          selected_indices = np.unique(top_k_indices.flatten())

    '''
    for i in range(pseudo_matrix.shape[0]):
        sample_cor = np.dot(pseudo_matrix[i,:],bulk_matrix.T)
        print(sample_cor)
        if sample_cor.max()>=cut_off:
            choose_index.append(sample_name[i])
    '''
    return cell_name[selected_indices]



def preprocessing(bulk_data,
                sc_adata,
                annotation_key,
                is_st:False,
                downsampling=True,
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
                n_sample_each_group=500,
                min_cells_each_group=100,
                cell_gap_each_group=100,
                group_number=5,
                filter_samples=False,
                python_path=None,
                **kwargs):
    """
    Preprocessing on bulk and sc adata, including following steps:\ 
        1. QC on bulk and sc adata.\ 
        2. Get common gene and common cell type.\  
        3. Get marker gene which is suitable for this dataset.\ 
        4. Normalization and scale.\ 
        5. Stimulation and batch effects.\ 
        6. NNLS to elimate gap between stimulated bulk and sc adata.\ 
        7. Transform gene expression value in input data.\ 

    Parameters
    ----------
    bulk_data: dataframe
        An :class:`~pandas.dataframe` containing the expression to normalization.
    sc_adata: anndata.AnnData
        An :class:`~anndata.AnnData` containing the expression to normalization.
    marker_data: 
        An :class:`~pandas.dataframe` which columns are cell types, rows are marker gene.
    annotation_key: string
        The `.obs` key where the single cell annotation is stored.: anndata.AnnData.
    project: string.
        The prefix of output file.
    out_dir: string, optional
        The path to store the output data.
    different_source: boolean, optional
        True for single cell and bulk data from the same sample, which means not executing batch effect.
        False for single cell and bulk data from the different samples, which means executing batch effect.
    cell_list: list, optional
        The list indicate the cell type names which need to take into consideration.
    scale_factors: int, optional
        The number of counts to normalize every observation to before computing profiles. If `None`, no normalization is performed. 
    trans_method: string, optional
        What transformation to apply to the expression before computing the profiles. 
        - "log": log(x+1)
        - `None`: no transformation
    save_figure: boolean, optional
        Whether save figures during preprocessing. eg. scatter plot of pca data.
    python_path: string, optional
        The path of python.exe for Giotto package.
    **kwargs: 
        Additional keyword arguments forwarded to
        :func:`~cytobulk.preprocessing.qc_bulk_sc`.
    

    Returns
    -------
    Returns the preprocessed bulk data (adata) containing the raw layer to store the raw data, 
    the batch_effected layer to fetch the expression after elimination of batch effect, and the transformed layer
    to fetch the expression value after linear transformation. Besides, adata.X stores the normalization data.
    adata.obsm['scale_factor'] saves the vector to normalize expression value.

    The stimualted bulk data (adata) using sc adata, and adata.X stores the normalization data. I also contains the
    normalization, batch_effected and transformed expression data. the cell_average_bulk key could get the bulk data
    stimulated with cell average expression. adata.obsm["sti_fraction"] stores the ground truth fraction.

    The preprocessed sc data (adata), which has .obsm["marker_gene"] to stores marker gene of valid cell types.

    """
    if  annotation_key not in sc_adata.obs:
        raise ValueError(f'The key {annotation_key!r} is not available in .obs!')
    # rename the cell type in sc_adata.obs
    if rename is not None:
        print("Start renaming cell type annotation")
        if not isinstance(rename,dict):
            raise ValueError(f"`A` can only be a dict but is a {rename.__class__}!")
        
        sc_adata.obs['curated_cell_type'] = sc_adata.obs.apply(lambda x: rename[x[annotation_key]] if x[annotation_key] in rename else "invalid", axis=1)
    else:
        sc_adata.obs.rename(columns={annotation_key: 'curated_cell_type'}, inplace=True)
        print("Finish renaming, the curated annotation could be found in sc_adata.obs['curated_cell_type']")
    annotation_key = 'curated_cell_type'
    print('================================================================================================================')
    print('------------------Start to check cell type annotation and quality control...------------------')
    sc_adata = sc_adata[sc_adata.obs[annotation_key]!="invalid",:]
    if is_st:
        sc_adata, bulk_adata, common_gene = qc_st_sc(bulk_data,sc_adata,out_dir=out_dir,save=save,dataset_name=dataset_name,**kwargs)
    else:
        sc_adata, bulk_adata, common_gene = qc_bulk_sc(bulk_data,sc_adata,out_dir=out_dir,save=save,dataset_name=dataset_name,**kwargs)

    common_cell = _check_cell_type(sc_adata,annotation_key,cell_list)
    print(common_cell)
    print('------------------Finish quality control------------------')

    if scale_factors is not None or trans_method is not None:
        sc_adata.layers["original"] = sc_adata.X
        bulk_adata.layers["original"] = bulk_adata.X
        sc_adata, bulk_adata = _normalization_data(bulk_adata,sc_adata,scale_factors,trans_method,
                                                save=False,project=dataset_name,out_dir=out_dir)

    # find informative genes.
    cell_type_counts = sc_adata.obs[annotation_key].value_counts()
    print("cell type after filtering:")
    print(cell_type_counts)
    if downsampling:
        print("start sampling")
        cell_types = sc_adata.obs[annotation_key].unique()
        sampled_indices = []
        for cell_type in cell_types:
            cell_type_indices = sc_adata.obs.index[sc_adata.obs[annotation_key] == cell_type].tolist()
            if 10*bulk_adata.shape[0]<3000:
                target_num=3000
            else:
                target_num=10*bulk_adata.shape[0]
            if len(cell_type_indices)<target_num:
                sampled_indices.extend(cell_type_indices)
            else:
                #sampled_indices.extend(_filter_cells(sc_adata[sc_adata.obs[annotation_key] == cell_type],bulk_adata))
                sampled_indices.extend(np.random.choice(cell_type_indices, target_num, replace=False))
                
        sc_adata = sc_adata[sampled_indices].copy()
        cell_type_counts = sc_adata.obs[annotation_key].value_counts()
        print("cell type after downsampling:")
        print(cell_type_counts)
        print("Finish")
    print('------------------Start to find hvg------------------')
    sc_adata = high_variable_gene(sc_adata,flavor='seurat')
    print(f'The number of valid genes of sc data is {sc_adata.shape[1]}')
    bulk_adata = high_variable_gene(bulk_adata,flavor='seurat')
    print(f'The number of valid genes of bulk data is {sc_adata.shape[1]}')
    print('------------------Finish------------------')
    common_gene = sc_adata.var.index.intersection(bulk_adata.var.index)
    # subset datasets
    print('Start to find vaild marker genes')
    if marker_list is not None:
        marker_list = common_gene.intersection(np.unique(np.array(marker_list)))
    else:
        marker_list = []
    sc_adata = sc_adata[:,common_gene]
    bulk_adata = bulk_adata[:,common_gene]
    '''
    if marker_list is not None:
        for i in db_marker.keys():
            flag=False
            if str(db_marker[i][0])!="nan":
                flag=True
            if not flag:
                db_gene=[]
            else:
                db_gene = np.array(db_marker[i])
            tmp = np.intersect1d(db_gene,common_gene)
            if len(db_marker) != len(tmp):
                db_marker[i] = tmp
    '''
    '''
    marker_dict,marker_gene,common_cell = _join_marker(bulk_adata,sc_adata,annotation_key,db_marker,common_cell,out_dir,dataset_name)
    '''
    marker_gene = _join_marker(bulk_adata,sc_adata,annotation_key,marker_list,out_dir,dataset_name)
    # debug
    print('Finish finding vaild marker genes')
    #marker_gene = np.intersect1d(np.array(marker_gene), sc_adata.var_names, assume_unique=False, return_indices=False)
    #bulk_adata = bulk_adata[:,marker_gene]
    bulk_adata = bulk_adata[:,marker_gene]
    sc_adata = sc_adata[sc_adata.obs[annotation_key].isin(common_cell),marker_gene]
    #sc_adata = sc_adata[sc_adata.obs[annotation_key].isin(common_cell),marker_gene]
    print(f'The number of valid single cell is {sc_adata.shape[0]}, valid sample is {bulk_adata.shape[0]},the number of valid genes is {sc_adata.shape[1]}')
    
    print('================================================================================================================')
    
    
    
    # training data simulation

    if is_st:
        if n_sample_each_group<800:
            n_sample_each_group=800
        pseudo_adata = utils.st_simulation(sc_adata, 
                bulk_adata,
                common_cell, 
                annotation_key = annotation_key,
                project=dataset_name, 
                out_dir=out_dir,
                n_sample_each_group=n_sample_each_group,
                min_cells_each_group=min_cells_each_group,
                cell_gap_each_group=cell_gap_each_group,
                group_number=group_number,
                average_ref=True,
                save=True,
                return_adata=True)
    else:
        if n_sample_each_group<400:
            n_sample_each_group=400
        pseudo_adata = utils.bulk_simulation(sc_adata,
                        bulk_adata,
                        common_cell, 
                        annotation_key = annotation_key,
                        project=dataset_name, 
                        out_dir=out_dir,
                        n_sample_each_group=n_sample_each_group,
                        min_cells_each_group=min_cells_each_group,
                        cell_gap_each_group=cell_gap_each_group,
                        group_number=group_number,
                        save=True,
                        return_adata=True)
    common_cell = pseudo_adata.obs.columns
    print("common_cell")
    print(common_cell)

    if save:
        out_dir = utils.check_paths(f'{out_dir}/filtered')
        pseudo_adata.write_h5ad(f"{out_dir}/pseudo_bulk_{dataset_name}.h5ad")
        sc_adata.write_h5ad(f"{out_dir}/sc_data_{dataset_name}.h5ad")
        bulk_adata.write_h5ad(f"{out_dir}/bulk_data_{dataset_name}.h5ad")
        with open(f"{out_dir}/cells_{dataset_name}.json", "w") as outfile: 
            json.dump(list(common_cell), outfile)

    return sc_adata, pseudo_adata, bulk_adata, common_cell, annotation_key

    
