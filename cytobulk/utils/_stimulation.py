
import time
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
from tqdm import tqdm
from numpy.random import choice
from ._read_data import check_paths
from ._math import normalization_cpm
from .. import preprocessing
from os.path import exists
from scipy.sparse import isspmatrix



def _get_stimulation(sc_data,meta_data,n_celltype,annotation_key,n_sample,n,round_th,project,set_missing=False):

    """
    Get stimulated expression data and cell type prop.
        
    Parameters
    ----------
    sc_adata : anndata.AnnData
        An :class:`~anndata.AnnData` containing the expression to stimulating bulk expression.
    n_celltype : int
        The cell type number used in stimulation.
    annotation_key : string
        String to save sc cell type information.
    n_sample : int
        The numbers of samples will be stimulated.
    n : int
        The number of cells included in each sample.
    round_th: int.
        The number to indicated the order of this round.
    
    
    Returns
    -------
    Returns the stimulated bulk adata.

    """

    cell_prop = np.random.dirichlet(np.ones(n_celltype), n_sample)
    print(cell_prop)
    #cell_prop[cell_prop < 1/n_celltype] = 0

    meta_index = meta_data[[annotation_key]]
    meta_index = meta_index.groupby(meta_data[annotation_key]).groups
    for key, value in meta_index.items():
        meta_index[key] = np.array(value)

    # scale prop value
    print(f'The number of samples is {cell_prop.shape[0]}, the number of cell types is {cell_prop.shape[1]}, cell number is {n}')
    if cell_prop.shape[1] > n:
        for j in range(int(cell_prop.shape[0])):
            cells = np.random.choice(np.arange(cell_prop.shape[1]), replace=False, size=cell_prop.shape[1]-n)
            cell_prop[j, cells] = 0
        cell_prop = cell_prop / np.sum(cell_prop, axis=1).reshape(-1, 1)

    if set_missing:
        for j in range(int(cell_prop.shape[0])):
            cells = np.random.choice(np.arange(cell_prop.shape[1]), replace=False, size=cell_prop.shape[1]-10)
            cell_prop[j, cells] = 0
        cell_prop = cell_prop / np.sum(cell_prop, axis=1).reshape(-1, 1)
        for i in range(int(cell_prop.shape[1])):
            indices = np.random.choice(np.arange(cell_prop.shape[0]), replace=False, size=int(cell_prop.shape[0] * 0.1))
            cell_prop[indices, i] = 0
        cell_prop = cell_prop / np.sum(cell_prop, axis=1).reshape(-1, 1)
        for i in range(int(cell_prop.shape[1])):
            indices = np.random.choice(np.arange(cell_prop.shape[0]), replace=False, size=int(cell_prop.shape[0] * 0.01))
            cell_prop[indices, :] = 0
            cell_prop[indices, i] = 1
        cell_prop = cell_prop / np.sum(cell_prop, axis=1).reshape(-1, 1)
    #refine the prop that can prop*cell_number < 1
    
    #cell_prop[cell_prop < 1/n] = 0
    #cell_prop = cell_prop / np.sum(cell_prop, axis=1).reshape(-1, 1)
    
    #genration of expression data and fraction data
    print('Start sampling...')
    sample = np.zeros((cell_prop.shape[0],sc_data.shape[0]))
    allcellname = meta_index.keys()
    print(meta_index.keys())
    cell_num = np.floor(n * cell_prop)
    print(cell_num)
    cell_prop_new = cell_num/ np.sum(cell_num, axis=1).reshape(-1, 1)
    for i, sample_prop in tqdm(enumerate(cell_num)):
        for j, cellname in enumerate(allcellname):
            select_index = choice(meta_index[cellname], size=int(sample_prop[j]), replace=True)
            sample[i] += sc_data.loc[:,select_index].sum(axis=1)
    sample = sample/n
    print("Sampling down")

    # generate a ref_adata
    cell_prop = pd.DataFrame(cell_prop_new,
            index=[f'Sample{str(n_sample * round_th + i)}_{project}' for i in range(n_sample)],
            columns=allcellname)
    sample_data = pd.DataFrame(sample,
        index=[f'Sample{str(n_sample * round_th + i)}_{project}' for i in range(n_sample)],
        columns=sc_data.index)
    

    return sample_data,cell_prop




def bulk_simulation(sc_adata,
                    cell_list,
                    annotation_key,
                    project,
                    out_dir,
                    n_sample_each_group=100,
                    min_cells_each_group=100,
                    cell_gap_each_group=100,
                    group_number=5,
                    rename_dict=None,
                    save=False,
                    return_adata=True):
    
    """
    Generation of bulk expression data with referenced sc adata. \  
        Total stimultated number = n_sample_each_group*group_number \ 
        The number of cells in different groups should be: min_cells_each_group, 
                                                            min_cells_each_group+cell_gap_each_group, 
                                                            min_cells_each_group+2*cell_gap_each_group,
                                                            min_cells_each_group+group_number*cell_gap_each_group
    Parameters
    ----------
    sc_adata : anndata.AnnData
        An :class:`~anndata.AnnData` containing the expression to stimulating bulk expression.
    cell_list : list
        The list of cell types for single cells, which will be used to generate simulated bulk data.
    project : string, optional
        The string used as the prefiex of the output file.
    out_dir : string, optional
        The path to save the output files.
    n_sample_each_group : int, optional
        The number of samples stimulated in each group.
    min_cells_each_group : int, optional
        Minimum number of cells contained in the sample.
    cell_gap_each_group : int, optional
        The gap in the number of cells between groups.
    group_number : int, optional
        The group number.
    rename_dict : dictionary, optional
        The dictionary to rename the cell types in sc adata.
    return_adata: 
        Return adata or dataframe.

    Returns
    -------
    Returns the stimulated bulk data and the corresponding cell type fraction.
    
    """

    start_t = time.perf_counter()

    print('')
    print('================================================================================================================')
    print('Start to stimulate the reference bulk expression data ...')
    if not isinstance(sc_adata.X, np.ndarray):
        sc_adata.X = sc_adata.X.toarray()

    start_t = time.perf_counter()

    n_celltype = len(cell_list)
    #subset sc data
    sub_sc_adata = sc_adata[sc_adata.obs[annotation_key].isin(cell_list),:].copy()
    #generate new data
    new_data = []
    new_prop = []
    sc_data = pd.DataFrame(sub_sc_adata.X,index=sub_sc_adata.obs_names,columns=sub_sc_adata.var_names).transpose()

    for i in range(group_number):
        ref_data,ref_prop = _get_stimulation(sc_data,
                                            sub_sc_adata.obs,
                                            n_celltype,
                                            annotation_key,
                                            n_sample_each_group,
                                            min_cells_each_group+i*cell_gap_each_group,
                                            i,
                                            project)
        new_data.append(ref_data)
        new_prop.append(ref_prop)

    ref_data = pd.concat(new_data)
    ref_prop = pd.concat(new_prop)
    
    if rename_dict is not None:
        ref_prop.rename(columns=rename_dict,inplace=True)

    '''
    training_prop.rename(columns={"B":"B cells","M1":"Macrophages M1","M2":"Macrophages M2","NK":"NK cells","CD4Tn":"T CD4 naive cells",
                                    "CD8Tex":"T CD8 exhausted cells","CD8Tem":"T CD8 effector memory cells","Monocyte":"Monocytes","Myofibroblasts":"Myofibroblasts",
                                    "Tprolif":"Tprolifs","Fibroblasts":"Fibroblasts","Endothelial":"Endothelial cells",
                                    "Mast":"Mast cells","Th17":"Th17 cells","Treg":"Tregs", "cDC1":"cDC cells","pDC":"pDC cells",
                                    "CD8Tn":"T CD8 naive cells","CD8Tcm":"T CD8 central memory cells","Plasma":"Plasma cells","cDC2":"cDC cells",
                                    "Epithelial":"Epithelial cells","Endothelial":"Endothelial cells","CD8Teff":"T CD8 effector cells",
                                    "Th1":"Th1 cells","CD8Tn":"T CD8 naive cells","Th2":"Th2 cells","CD4Tconv":"T CD4 conventional cells"},inplace=True)
    '''
    
    print(f'Time to generate bulk data: {round(time.perf_counter() - start_t, 2)} seconds')



    if save:
        # check out path
        reference_out_dir = check_paths(out_dir+'/reference_bulk_data')
        print('Saving stimulated data')
        ref_data.to_csv(reference_out_dir+f"/{project}_stimulated_bulk.txt",sep='\t')
        ref_prop.to_csv(reference_out_dir+f"/{project}_stimulated_prop.txt",sep='\t')
    print('Finish bulk stimulation.')
    print('================================================================================================================')

    if not return_adata:
        return ref_data,ref_prop
    
    adata = sc.AnnData(ref_data)
    adata.obs = ref_prop
    return adata


def st_simulation(sc_adata,
                cell_list,
                annotation_key,
                project,
                out_dir,
                n_sample_each_group=1000,
                min_cells_each_group=8,
                cell_gap_each_group=1,
                group_number=5,
                rename_dict=None,
                save=False,
                return_adata=True):
    
    """
    Generation of bulk expression data with referenced sc adata. \  
        Total stimultated number = n_sample_each_group*group_number \ 
        The number of cells in different groups should be: min_cells_each_group, 
                                                            min_cells_each_group+cell_gap_each_group, 
                                                            min_cells_each_group+2*cell_gap_each_group,
                                                            min_cells_each_group+group_number*cell_gap_each_group
    Parameters
    ----------
    sc_adata : anndata.AnnData
        An :class:`~anndata.AnnData` containing the expression to stimulating bulk expression.
    cell_list : list
        The list of cell types for single cells, which will be used to generate simulated bulk data.
    project : string, optional
        The string used as the prefiex of the output file.
    out_dir : string, optional
        The path to save the output files.
    n_sample_each_group : int, optional
        The number of samples stimulated in each group.
    min_cells_each_group : int, optional
        Minimum number of cells contained in the sample.
    cell_gap_each_group : int, optional
        The gap in the number of cells between groups.
    group_number : int, optional
        The group number.
    rename_dict : dictionary, optional
        The dictionary to rename the cell types in sc adata.
    return_adata: 
        Return adata or dataframe.

    Returns
    -------
    Returns the stimulated bulk data and the corresponding cell type fraction.
    
    """
    reference_out_dir = check_paths(out_dir+'/reference_st_data')
    if exists(f"{reference_out_dir}/filtered_{project}_st.h5ad"):
        adata = sc.read_h5ad(f"{reference_out_dir}/filtered_{project}_st.h5ad")
        print(f'{reference_out_dir}/filtered_{project}_st.h5ad already exists, skipping simulation.')
        return adata

    else:
        start_t = time.perf_counter()

        print('')
        print('================================================================================================================')
        print('Start to stimulate the reference bulk expression data ...')

        #if not isinstance(sc_adata.X, np.ndarray):
            #sc_adata.X = sc_adata.X.toarray()

        start_t = time.perf_counter()

        n_celltype = len(cell_list)
        #subset sc data
        sub_sc_adata = sc_adata[sc_adata.obs[annotation_key].isin(cell_list),:].copy()
        #generate new data
        if isspmatrix(sub_sc_adata.X):
            sub_sc_adata.X = sub_sc_adata.X.todense()
        new_data = []
        new_prop = []
        print(sub_sc_adata)
        sc_data = pd.DataFrame(sub_sc_adata.X,index=sub_sc_adata.obs_names,columns=sub_sc_adata.var_names).transpose()

        for i in range(group_number):
            ref_data,ref_prop = _get_stimulation(sc_data,
                                                sub_sc_adata.obs,
                                                n_celltype,
                                                annotation_key,
                                                n_sample_each_group,
                                                min_cells_each_group+i*cell_gap_each_group,
                                                i,
                                                project,
                                                set_missing=False)
            new_data.append(ref_data)
            new_prop.append(ref_prop)

        ref_data = pd.concat(new_data)
        ref_prop = pd.concat(new_prop)
        
        if rename_dict is not None:
            ref_prop.rename(columns=rename_dict,inplace=True) 
        print(f'Time to generate st data: {round(time.perf_counter() - start_t, 2)} seconds')


        print('Finish st stimulation.')
        print('================================================================================================================')

    if not return_adata:
        if save:
            # check out path
            reference_out_dir = check_paths(out_dir+'/reference_st_data')
            print('Saving stimulated data')
            ref_data.to_csv(reference_out_dir+f"/{project}_stimulated_st.txt",sep='\t')
            ref_prop.to_csv(reference_out_dir+f"/{project}_stimulated_st.txt",sep='\t')
        return ref_data,ref_prop
    else:
        adata = sc.AnnData(ref_data)
        adata.obs = ref_prop
        if save:
            # check out path
            reference_out_dir = check_paths(out_dir+'/reference_st_data')
            print('Saving stimulated data')
            ref_data.to_csv(reference_out_dir+f"/{project}_stimulated_st_exp.txt",sep='\t')
            ref_prop.to_csv(reference_out_dir+f"/{project}_stimulated_st_prop.txt",sep='\t')
            adata.write_h5ad(f"{reference_out_dir}/filtered_{project}_st.h5ad") 

        return adata



def st_simulation_case(sc_adata,
                    cell_list,
                    annotation_key,
                    project,
                    out_dir,
                    n_sample_each_group=100,
                    min_cells_each_group=6,
                    cell_gap_each_group=1,
                    group_number=5,
                    rename_dict=None,
                    save=True,
                    scale_factors=10000,
                    trans_method="log",
                    return_adata=True):
    
    """
    Generation of bulk expression data with referenced sc adata.
        Total stimultated number = n_sample_each_group*group_number\ 
        The number of cells in different groups should be: min_cells_each_group, 
                                                            min_cells_each_group+cell_gap_each_group, 
                                                            min_cells_each_group+2*cell_gap_each_group,
                                                            min_cells_each_group+group_number*cell_gap_each_group
    Parameters
    ----------
    sc_adata : anndata.AnnData
        An :class:`~anndata.AnnData` containing the expression to stimulating st expression.
    cell_list : list
        The list of cell types for single cells, which will be used to generate simulated bulk data.
    project: string, optional
        The string used as the prefiex of the output file.
    out_dir : string, optional
        The path to save the output files.
    n_sample_each_group : int, optional
        The number of samples stimulated in each group.
    min_cells_each_group : int, optional
        Minimum number of cells contained in the sample.
    cell_gap_each_group : int, optional
        The gap in the number of cells between groups.
    group_number : int, optional
        The group number.
    rename_dict : dictionary, optional
        The dictionary to rename the cell types in sc adata.
    return_adata: 
        Return adata or dataframe.

    Returns
    -------
    Returns the stimulated bulk data and the corresponding cell type fraction.

    """

    start_t = time.perf_counter()

    print('')
    print('================================================================================================================')
    if rename_dict is not None:
        print("Start renaming cell type annotation")
        if not isinstance(rename_dict,dict):
            raise ValueError(f"`A` can only be a dict but is a {rename_dict.__class__}!")
        meta = sc_adata.obs
        meta['curated_cell_type'] = meta.apply(lambda x: rename_dict[x[annotation_key]] if x[annotation_key] in rename_dict else "invalid", axis=1)
        sc_adata.obs['curated_cell_type']=meta['curated_cell_type']
        annotation_key = 'curated_cell_type'
        print("Finish renaming, the curated annotation could be found in sc_adata.obs['curated_cell_type']")

    print('================================================================================================================')
    print('Start to check cell type annotation and quality control...')
    sc_adata = sc_adata[sc_adata.obs[annotation_key]!="invalid",:]


    start_t = time.perf_counter()
    print(np.unique(sc_adata.obs[annotation_key]))
    sc_adata=preprocessing.qc_sc(sc_adata)
    print(np.unique(sc_adata.obs[annotation_key]))
    #sc_adata = normalization_cpm(sc_adata,scale_factors,trans_method)
    n_celltype = len(cell_list)
    #subset sc data
    print('Start to stimulate the reference st expression data ...')
    #generate new data
    new_data = []
    new_prop = []
    if not isinstance(sc_adata.X, np.ndarray):
        sc_data = sc_adata.X.toarray()
    sc_data = pd.DataFrame(sc_data,index=sc_adata.obs_names,columns=sc_adata.var_names).transpose()

    for i in range(group_number):
        ref_data,ref_prop = _get_stimulation(sc_data,
                                            sc_adata.obs,
                                            n_celltype,
                                            annotation_key,
                                            n_sample_each_group,
                                            min_cells_each_group+i*cell_gap_each_group,
                                            i,
                                            project,
                                            set_missing=False)
        new_data.append(ref_data)
        new_prop.append(ref_prop)

    ref_data = pd.concat(new_data)
    ref_prop = pd.concat(new_prop)

    print(f'Time to generate st data: {round(time.perf_counter() - start_t, 2)} seconds')

    print('Finish st stimulation.')
    print('================================================================================================================')

    if not return_adata:
        if save:
            # check out path
            reference_out_dir = check_paths(out_dir+'/reference_st_data')
            print('Saving stimulated data')
            ref_data.to_csv(reference_out_dir+f"/{project}_stimulated_st.txt",sep='\t')
            ref_prop.to_csv(reference_out_dir+f"/{project}_stimulated_st.txt",sep='\t')
        return ref_data,ref_prop
    else:
        adata = sc.AnnData(ref_data)
        adata.obs = ref_prop
        if save:
            # check out path
            reference_out_dir = check_paths(out_dir+'/reference_st_data')
            print('Saving stimulated data')
            ref_data.to_csv(reference_out_dir+f"/{project}_stimulated_st_exp_{min_cells_each_group}.txt",sep='\t')
            ref_prop.to_csv(reference_out_dir+f"/{project}_stimulated_st_prop_{min_cells_each_group}.txt",sep='\t')
            adata.write_h5ad(f"{reference_out_dir}/stimulated_{project}_st_{min_cells_each_group}.h5ad") 

        return adata


def bulk_simulation_case(sc_adata,
                        cell_list,
                        annotation_key,
                        project,
                        out_dir,
                        n_sample_each_group=100,
                        min_cells_each_group=100,
                        cell_gap_each_group=100,
                        group_number=5,
                        rename_dict=None,
                        save=True,
                        scale_factors=100000,
                        trans_method="log",
                        return_adata=False):
    
    """
    Generation of bulk expression data with referenced sc adata.
        Total stimultated number = n_sample_each_group*group_number\ 
        The number of cells in different groups should be: min_cells_each_group, 
                                                            min_cells_each_group+cell_gap_each_group, 
                                                            min_cells_each_group+2*cell_gap_each_group,
                                                            min_cells_each_group+group_number*cell_gap_each_group
    Parameters
    ----------
    sc_adata : anndata.AnnData
        An :class:`~anndata.AnnData` containing the expression to stimulating bulk expression.
    cell_list : list
        The list of cell types for single cells, which will be used to generate simulated bulk data.
    project: string, optional
        The string used as the prefiex of the output file.
    out_dir : string, optional
        The path to save the output files.
    n_sample_each_group : int, optional
        The number of samples stimulated in each group.
    min_cells_each_group : int, optional
        Minimum number of cells contained in the sample.
    cell_gap_each_group : int, optional
        The gap in the number of cells between groups.
    group_number : int, optional
        The group number.
    rename_dict : dictionary, optional
        The dictionary to rename the cell types in sc adata.
    return_adata: 
        Return adata or dataframe.

    Returns
    -------
    Returns the stimulated bulk data and the corresponding cell type fraction.

    """

    start_t = time.perf_counter()

    print('')
    print('================================================================================================================')
    if rename_dict is not None:
        print("Start renaming cell type annotation")
        if not isinstance(rename_dict,dict):
            raise ValueError(f"`A` can only be a dict but is a {rename_dict.__class__}!")
        meta = sc_adata.obs
        meta['curated_cell_type'] = meta.apply(lambda x: rename_dict[x[annotation_key]] if x[annotation_key] in rename_dict else "invalid", axis=1)
        annotation_key = 'curated_cell_type'
        print("Finish renaming, the curated annotation could be found in sc_adata.obs['curated_cell_type']")

    print('================================================================================================================')
    print('Start to check cell type annotation and quality control...')
    sc_adata = sc_adata[sc_adata.obs[annotation_key]!="invalid",:]
    


    start_t = time.perf_counter()

    sc_adata= preprocessing.qc_sc(sc_adata)
    #sc_adata = normalization_cpm(sc_adata,scale_factors,trans_method)
    n_celltype = len(cell_list)
    #subset sc data
    print('Start to stimulate the reference bulk expression data ...')
    #generate new data
    new_data = []
    new_prop = []
    if not isinstance(sc_adata.X, np.ndarray):
        sc_data = sc_adata.X.toarray()
    sc_data = pd.DataFrame(sc_data,index=sc_adata.obs_names,columns=sc_adata.var_names).transpose()

    for i in range(group_number):
        ref_data,ref_prop = _get_stimulation(sc_data,
                                            sc_adata.obs,
                                            n_celltype,
                                            annotation_key,
                                            n_sample_each_group,
                                            min_cells_each_group+i*cell_gap_each_group,
                                            i,
                                            project)
        new_data.append(ref_data)
        new_prop.append(ref_prop)

    ref_data = pd.concat(new_data)
    ref_prop = pd.concat(new_prop)
    
    print(f'Time to generate bulk data: {round(time.perf_counter() - start_t, 2)} seconds')



    if save:
        # check out path
        reference_out_dir = check_paths(out_dir+'/reference_bulk_data')
        print('Saving stimulated data')
        ref_data.to_csv(reference_out_dir+f"/{project}_stimulated_bulk.txt",sep='\t')
        ref_prop.to_csv(reference_out_dir+f"/{project}_stimulated_prop.txt",sep='\t')
    print('Finish bulk stimulation.')
    print('================================================================================================================')

    if not return_adata:
        return ref_data,ref_prop
    
    adata = sc.AnnData(ref_data)
    adata.obs = ref_prop
    return adata