
import time
import pandas as pd
import numpy as np
import scanpy as sc
import random
import math
from numpy.random import choice
from ._read_data import check_paths
from ._utils import compute_cluster_averages
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

    #cell_prop[cell_prop < 1/n_celltype] = 0

    meta_index = meta_data[[annotation_key]]
    meta_index = meta_index.groupby(meta_data[annotation_key]).groups
    for key, value in meta_index.items():
        meta_index[key] = np.array(value)
    
    # scale prop value
    if cell_prop.shape[1] > n:
        for j in range(int(cell_prop.shape[0])):
            cells = np.random.choice(np.arange(cell_prop.shape[1]), replace=False, size=cell_prop.shape[1]-n)
            cell_prop[j, cells] = 0
        cell_prop = cell_prop / np.sum(cell_prop, axis=1).reshape(-1, 1)

    if set_missing:
        for i in range(int(cell_prop.shape[1])):
            indices = np.random.choice(np.arange(cell_prop.shape[0]), replace=False, size=int(cell_prop.shape[0] * 0.1))
            cell_prop[indices, i] = 0
        cell_prop = cell_prop / np.sum(cell_prop, axis=1).reshape(-1, 1)
    #refine the prop that can prop*cell_number < 1
    
    #cell_prop[cell_prop < 1/n] = 0
    #cell_prop = cell_prop / np.sum(cell_prop, axis=1).reshape(-1, 1)
    
    #genration of expression data and fraction data

    sample = np.zeros((cell_prop.shape[0],sc_data.shape[0]))
    allcellname = meta_index.keys()
    cell_num = np.floor(n * cell_prop)
    cell_prop_new = cell_num/ np.sum(cell_num, axis=1).reshape(-1, 1)
    for i, sample_prop in enumerate(cell_num):
        for j, cellname in enumerate(allcellname):
            select_index = choice(meta_index[cellname], size=int(sample_prop[j]), replace=True)
            sample[i] += sc_data.loc[:,select_index].sum(axis=1)
    sample = sample/n


    # generate a ref_adata
    cell_prop = pd.DataFrame(cell_prop_new,
            index=[f'Sample{str(n_sample * round_th + i)}_{project}' for i in range(n_sample)],
            columns=allcellname)
    sample_data = pd.DataFrame(sample,
        index=[f'Sample{str(n_sample * round_th + i)}_{project}' for i in range(n_sample)],
        columns=sc_data.index)
    

    return sample_data,cell_prop
    
    
def _get_prop_sample_bulk(sc_data,meta_data,cell_composition,n_celltype,cell_specific,annotation_key,n_sample,n,round_th,project,set_missing=False):

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
    meta_index = meta_data[[annotation_key]]
    meta_index = meta_index.groupby(meta_data[annotation_key]).groups
    allcellname = list(meta_index.keys())

    selected_index = allcellname.index(cell_specific)
    all_cell_num = len(meta_index[cell_specific])
    if all_cell_num>=1000:
      n_sample = n_sample*2
    if all_cell_num<=30:
      n_sample = math.ceil(n_sample/2)

    cell_prop = np.zeros((n_sample,n_celltype))

    get_prop = np.random.dirichlet(np.ones(len(cell_composition)), n_sample)
    for i in range(n_sample):
        cell_prop[i,cell_composition]=get_prop[i,:]
      

    #cell_prop[cell_prop < 1/n_celltype] = 0


    for key, value in meta_index.items():
        meta_index[key] = np.array(value)
    # scale prop value
    #if cell_prop.shape[1] > 5:
      #cut_num=[3,4,5]
    #else:
      #cut_num=np.arange(3,cell_prop.shape[1]+1,1) 
    #for sample in range(int(cell_prop.shape[0])):
        #for num in cut_num:
        #cells = np.random.choice(np.arange(cell_prop.shape[1]), replace=False, size=cell_prop.shape[1]-5)
        #cell_prop[sample, cells] = 0
        #cell_prop[:,selected_index]+=0.05
    for i in range(int(cell_prop.shape[1])):
        indices = np.random.choice(np.arange(cell_prop.shape[0]), replace=False, size=int(cell_prop.shape[0] * 0.1))
        cell_prop[indices, i] = 0
    cell_prop = cell_prop / np.sum(cell_prop, axis=1).reshape(-1, 1)
    for i in range(int(cell_prop.shape[0])):
        cell_prop[i,selected_index]=cell_prop[i,selected_index]+random.uniform(0, 1)
    cell_prop = cell_prop / np.sum(cell_prop, axis=1).reshape(-1, 1)
    sample = np.zeros((cell_prop.shape[0],sc_data.shape[0]))
    cell_num = np.floor(n * cell_prop)
    cell_prop_new = cell_num/ np.sum(cell_num, axis=1).reshape(-1, 1)
    for i, sample_prop in enumerate(cell_num):
        for j, cellname in enumerate(allcellname):
            select_index = choice(meta_index[cellname], size=int(sample_prop[j]), replace=True)
            sample[i] += sc_data.loc[:,select_index].sum(axis=1)
    sample = sample/n


    # generate a ref_adata
    cell_prop = pd.DataFrame(cell_prop_new,
            index=[f'Sample{str(n_sample * round_th + i)}_{project}' for i in range(n_sample)],
            columns=allcellname)
    sample_data = pd.DataFrame(sample,
        index=[f'Sample{str(n_sample * round_th + i)}_{project}' for i in range(n_sample)],
        columns=sc_data.index)

    return sample_data,cell_prop



def _get_prop_sample_sti(sc_data,meta_data,cell_composition,n_celltype,cell_specific,annotation_key,n_sample,n,round_th,project,set_missing=False):

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
    meta_index = meta_data[[annotation_key]]
    meta_index = meta_index.groupby(meta_data[annotation_key]).groups
    allcellname = list(meta_index.keys())

    selected_index = allcellname.index(cell_specific)
    all_cell_num = len(meta_index[cell_specific])
    if all_cell_num>=1000:
      n_sample = n_sample*2
    if all_cell_num<=30:
      n_sample = math.ceil(n_sample/2)

    cell_prop = np.zeros((n_sample,n_celltype))

    get_prop = np.random.dirichlet(np.ones(len(cell_composition)), n_sample)
    if len(cell_composition) > 6:
        get_prop = np.random.dirichlet(np.ones(6), n_sample)
        for j in range(int(get_prop.shape[0])):
            cell_chosen = np.random.choice(cell_composition, replace=False, size=6).tolist()
            if selected_index not in cell_chosen:
                cell_chosen.pop()
                cell_chosen.append(selected_index)
            cell_prop[j, cell_chosen] = get_prop[j,:]
    else:
        for i in range(n_sample):
            cell_prop[i,cell_composition]=get_prop[i,:]
      

    #cell_prop[cell_prop < 1/n_celltype] = 0


    for key, value in meta_index.items():
        meta_index[key] = np.array(value)
    # scale prop value

    #if cell_prop.shape[1] > 5:
      #cut_num=[3,4,5]
    #else:
      #cut_num=np.arange(3,cell_prop.shape[1]+1,1) 
    #for sample in range(int(cell_prop.shape[0])):
        #for num in cut_num:
        #cells = np.random.choice(np.arange(cell_prop.shape[1]), replace=False, size=cell_prop.shape[1]-5)
        #cell_prop[sample, cells] = 0
        #cell_prop[:,selected_index]+=0.05
    
    cell_prop[:,selected_index]=cell_prop[:,selected_index]+random.uniform(1, 2)
    cell_prop = cell_prop / np.sum(cell_prop, axis=1).reshape(-1, 1)
    sample = np.zeros((cell_prop.shape[0],sc_data.shape[0]))
    cell_num = np.floor(n * cell_prop)
    cell_prop_new = cell_num/ np.sum(cell_num, axis=1).reshape(-1, 1)
    for i, sample_prop in enumerate(cell_num):
        for j, cellname in enumerate(allcellname):
            select_index = choice(meta_index[cellname], size=int(sample_prop[j]), replace=True)
            sample[i] += sc_data.loc[:,select_index].sum(axis=1)
    sample = sample/n

    # generate a ref_adata
    cell_prop = pd.DataFrame(cell_prop_new,
            index=[f'Sample{str(n_sample * round_th + i)}_{project}' for i in range(n_sample)],
            columns=allcellname)
    sample_data = pd.DataFrame(sample,
        index=[f'Sample{str(n_sample * round_th + i)}_{project}' for i in range(n_sample)],
        columns=sc_data.index)

    return sample_data,cell_prop
    

def bulk_simulation(sc_adata,
                    bulk_adata,
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


    start_t = time.perf_counter()

    n_celltype = len(cell_list)
    #subset sc data
    sub_sc_adata = sc_adata[sc_adata.obs[annotation_key].isin(cell_list),:].copy()
    #generate new data
    new_data = []
    new_prop = []
    if isspmatrix(sub_sc_adata.X):
          sub_sc_adata.X = sub_sc_adata.X.todense()
    from sklearn.metrics.pairwise import cosine_similarity
    from collections import Counter
    average_cell_exp = compute_cluster_averages(sub_sc_adata,annotation_key,cell_list,out_dir=out_dir,project=project,save=True).T
    if not isinstance(bulk_adata.X, np.ndarray):
        bulk_data = pd.DataFrame(bulk_adata.X.todense(),index=bulk_adata.obs_names,columns=bulk_adata.var_names)
    else:
        bulk_data = pd.DataFrame(bulk_adata.X,index=bulk_adata.obs_names,columns=bulk_adata.var_names)
    similarity_matrix = cosine_similarity(bulk_data.values,average_cell_exp.values)
    all_most_index=np.argsort(-similarity_matrix, axis=1)[:, 0].flatten()
    most_sim_index = np.unique(all_most_index)
    most_sim_cell = [average_cell_exp.index.tolist()[x] for i, x in enumerate(all_most_index)]
    cell_prop = Counter(most_sim_cell)
    for cell_name in cell_prop.keys():
        cell_prop[cell_name] = cell_prop[cell_name]/bulk_data.shape[0]
    selected_index = np.argsort(-similarity_matrix, axis=1)
    selected_sim_index = np.unique(np.argsort(-similarity_matrix, axis=1).flatten())
    rare_cell_list = np.setdiff1d(selected_sim_index,most_sim_index,False)
    all_cells_names = np.array(average_cell_exp.index.tolist())[selected_sim_index]
    sample_cell_composition =  dict(enumerate(selected_index))
    
    sc_data = pd.DataFrame(sub_sc_adata.X,index=sub_sc_adata.obs_names,columns=sub_sc_adata.var_names).transpose()
    
    for i in range(group_number):
        for j in range(bulk_data.shape[0]):
            selected_celltype = sample_cell_composition[j]
            cells = np.array(average_cell_exp.index.tolist())[selected_celltype[0]]
            change_fold = cell_prop[cells]/(1/len(all_cells_names))
            if change_fold>=1.5:
                sti_num=round(n_sample_each_group*((1/change_fold)/bulk_data.shape[0]))
                if sti_num<3:
                    sti_num=3
            elif change_fold<=0.05:
                sti_num=round(n_sample_each_group*(((1/10)/change_fold)/bulk_data.shape[0]))
            elif change_fold<=0.1:
                sti_num=round(n_sample_each_group*(((1/4)/change_fold)/bulk_data.shape[0]))
            elif change_fold<=0.5:
                sti_num=round(n_sample_each_group*(((1/2)/change_fold)/bulk_data.shape[0]))
            else:
                sti_num=round(n_sample_each_group*(1/bulk_data.shape[0]))

            ref_data,ref_prop = _get_prop_sample_bulk(sc_data,
                                                    sub_sc_adata.obs,
                                                    selected_celltype,
                                                    n_celltype,
                                                    cells,
                                                    annotation_key,
                                                    sti_num,
                                                    min_cells_each_group+i*cell_gap_each_group,
                                                    i,
                                                    project,
                                                    set_missing=False)
            new_data.append(ref_data)
            new_prop.append(ref_prop)
                    
            '''
            selected_celltype = sample_cell_composition[j]
            cells = np.array(average_cell_exp.index.tolist())[selected_celltype[0]]
            sti_num = int(n_sample_each_group*((1/2)/bulk_data.shape[0]))
            ref_data,ref_prop = _get_prop_sample_bulk(sc_data,
                                                    sub_sc_adata.obs,
                                                    selected_celltype,
                                                    n_celltype,
                                                    cells,
                                                    annotation_key,
                                                    sti_num,
                                                    min_cells_each_group+i*cell_gap_each_group,
                                                    i,
                                                    project,
                                                    set_missing=False)
            new_data.append(ref_data)
            new_prop.append(ref_prop)
            '''
        for rare_cell in rare_cell_list:
            if int(n_sample_each_group*((1/len(all_cells_names)/10)))<len(all_cells_names):
                sti_num=len(all_cells_names)
            else:
                sti_num = int(n_sample_each_group*((1/len(all_cells_names)/10)))
            cells = np.array(average_cell_exp.index.tolist())[rare_cell]
            ref_data,ref_prop = _get_prop_sample_bulk(sc_data,
                                sub_sc_adata.obs,
                                selected_sim_index,
                                n_celltype,
                                cells,
                                annotation_key,
                                sti_num,
                                min_cells_each_group+i*cell_gap_each_group,
                                i,
                                project,
                                set_missing=False)
            new_data.append(ref_data)
            new_prop.append(ref_prop)
        
        ref_data,ref_prop = _get_stimulation(sc_data,
                          sub_sc_adata.obs,
                          n_celltype,
                          annotation_key,
                          int(n_sample_each_group*0.2),
                          min_cells_each_group+i*cell_gap_each_group,
                          i,
                          project,
                          set_missing=True)
        new_data.append(ref_data)
        new_prop.append(ref_prop)
    ref_data = pd.concat(new_data)
    ref_prop = pd.concat(new_prop)
    ref_data = pd.DataFrame(ref_data.values,
                index=[f'Sample{str(i)}_{project}' for i in range(ref_data.shape[0])],
                columns=ref_data.columns)
    ref_prop = pd.DataFrame(ref_prop.values,
                index=[f'Sample{str(i)}_{project}' for i in range(ref_data.shape[0])],
                columns=ref_prop.columns)

        
    if rename_dict is not None:
        ref_prop.rename(columns=rename_dict,inplace=True)


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
                st_adata,
                cell_list,
                annotation_key,
                project,
                out_dir,
                n_sample_each_group=1000,
                min_cells_each_group=8,
                cell_gap_each_group=1,
                group_number=5,
                rename_dict=None,
                average_ref=False,
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
        if not isinstance(sub_sc_adata.X, np.ndarray):
            sub_sc_adata.X = sub_sc_adata.X.toarray()
        new_data = []
        new_prop = []
        sc_data = pd.DataFrame(sub_sc_adata.X,index=sub_sc_adata.obs_names,columns=sub_sc_adata.var_names).transpose()
        if not average_ref:
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
        else:
            from sklearn.metrics.pairwise import cosine_similarity
            from collections import Counter
            average_cell_exp = compute_cluster_averages(sub_sc_adata,annotation_key,cell_list,out_dir=out_dir,project=project,save=True).T
            if not isinstance(st_adata.X, np.ndarray):
                st_data = pd.DataFrame(st_adata.X.todense(),index=st_adata.obs_names,columns=st_adata.var_names)
            else:
                st_data = pd.DataFrame(st_adata.X,index=st_adata.obs_names,columns=st_adata.var_names)
            similarity_matrix = cosine_similarity(st_data.values,average_cell_exp.values)
            all_most_index=np.argsort(-similarity_matrix, axis=1)[:, 0].flatten()
            most_sim_index = np.unique(all_most_index)
            most_sim_cell = [average_cell_exp.index.tolist()[x] for i, x in enumerate(all_most_index)]
            cell_prop = Counter(most_sim_cell)
            for cell_name in cell_prop.keys():
                cell_prop[cell_name] = cell_prop[cell_name]/st_data.shape[0]
            selected_index = np.argsort(-similarity_matrix, axis=1)[:, :5]
            selected_sim_index = np.unique(np.argsort(-similarity_matrix, axis=1)[:,:5].flatten())
            rare_cell_list = np.setdiff1d(selected_sim_index,most_sim_index,False)
            all_cells_names = np.array(average_cell_exp.index.tolist())[selected_sim_index]
            sample_cell_composition =  dict(enumerate(selected_index))

            for i in range(group_number):
                for j in range(st_data.shape[0]):
                    selected_celltype = sample_cell_composition[j]
                    cells = np.array(average_cell_exp.index.tolist())[selected_celltype[0]]
                    '''
                    if cell_prop[cells]>0.6:
                        sti_num=1
                    elif cell_prop[cells]>(6/len(all_cells_names)) or cell_prop[cells]>0.3:
                        sti_num=2
                    elif cell_prop[cells]>(4/len(all_cells_names)) or cell_prop[cells]>0.2:
                        sti_num=3
                    elif cell_prop[cells]>(2/len(all_cells_names)):
                        sti_num=4
                    elif cell_prop[cells]<0.04:
                        sti_num=8
                    else:
                        sti_num=5
                    '''
                    change_fold = cell_prop[cells]/(1/len(all_cells_names))
                    print(change_fold)
                    if change_fold>=1.5:
                        sti_num=round(n_sample_each_group*((1/change_fold)/st_data.shape[0]))
                        if sti_num<3:
                            sti_num=3
                    elif change_fold<=0.05:#1/10
                        sti_num=round(n_sample_each_group*(((0.1/change_fold)/st_data.shape[0])))
                    elif change_fold<=0.1:#1/4
                        sti_num=round(n_sample_each_group*(((0.25/change_fold)/st_data.shape[0])))
                    elif change_fold<=0.5:
                        sti_num=round(n_sample_each_group*(((0.5/change_fold)/st_data.shape[0])))
                    else:
                        sti_num=round(n_sample_each_group*(1/st_data.shape[0]))

                    ref_data,ref_prop = _get_prop_sample_sti(sc_data,
                                                            sub_sc_adata.obs,
                                                            selected_celltype,
                                                            n_celltype,
                                                            cells,
                                                            annotation_key,
                                                            sti_num,
                                                            min_cells_each_group+i*cell_gap_each_group,
                                                            i,
                                                            project,
                                                            set_missing=False)
                    new_data.append(ref_data)
                    new_prop.append(ref_prop)
                    
                    #if len(np.intersect1d(rare_cell_list,np.array(selected_celltype)))>0:
                for rare_cell in rare_cell_list:
                    cells = np.array(average_cell_exp.index.tolist())[rare_cell]
                    ref_data,ref_prop = _get_prop_sample_sti(sc_data,
                                        sub_sc_adata.obs,
                                        selected_sim_index,
                                        n_celltype,
                                        cells,
                                        annotation_key,
                                        int(n_sample_each_group*0.02),
                                        min_cells_each_group+i*cell_gap_each_group,
                                        i,
                                        project,
                                        set_missing=False)
                    new_data.append(ref_data)
                    new_prop.append(ref_prop)
                for cell_keys in cell_prop.keys():
                    if cell_prop[cell_keys]<(1/len(all_cells_names)/2):
                        if int(n_sample_each_group*((1/len(all_cells_names)/10)))<10:
                            sti_num=10
                        else:
                            sti_num = int(n_sample_each_group*((1/len(all_cells_names)/10)))
                        ref_data,ref_prop = _get_prop_sample_sti(sc_data,
                                        sub_sc_adata.obs,
                                        selected_sim_index,
                                        n_celltype,
                                        cell_keys,
                                        annotation_key,
                                        sti_num,
                                        min_cells_each_group+i*cell_gap_each_group,
                                        i,
                                        project,
                                        set_missing=False)
                        new_data.append(ref_data)
                        new_prop.append(ref_prop)

            ref_data = pd.concat(new_data)
            ref_prop = pd.concat(new_prop)
            ref_data = pd.DataFrame(ref_data.values,
                        index=[f'Sample{str(i)}_{project}' for i in range(ref_data.shape[0])],
                        columns=ref_data.columns)
            ref_prop = pd.DataFrame(ref_prop.values,
                        index=[f'Sample{str(i)}_{project}' for i in range(ref_data.shape[0])],
                        columns=ref_prop.columns)
            nonzero_columns = ref_prop.any()
            ref_prop = ref_prop.loc[:, nonzero_columns]
            
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
    ref_prop = ref_prop.loc[(ref_prop!=0).any(axis=0)]
    
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
    return 
