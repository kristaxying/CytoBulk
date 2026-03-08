
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
from scipy.sparse import issparse
from collections import Counter



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
    #cell_prop[cell_prop < 0.02] = 0
    cell_prop = cell_prop / np.sum(cell_prop, axis=1).reshape(-1, 1)
    meta_index = meta_data[[annotation_key]]
    meta_index = meta_index.groupby(meta_data[annotation_key]).groups
    for key, value in meta_index.items():
        meta_index[key] = np.array(value)
    
    # scale prop value
    '''
    if cell_prop.shape[1] > n:
        for j in range(int(cell_prop.shape[0])):
            cells = np.random.choice(np.arange(cell_prop.shape[1]), replace=False, size=cell_prop.shape[1]-n)
            cell_prop[j, cells] = 0
        cell_prop = cell_prop / np.sum(cell_prop, axis=1).reshape(-1, 1)
    '''
    '''

    if set_missing:
        for i in range(int(cell_prop.shape[1])):
            indices = np.random.choice(np.arange(cell_prop.shape[0]), replace=False, size=int(cell_prop.shape[0] * 0.1))
            cell_prop[indices, i] = 0
        cell_prop = cell_prop / np.sum(cell_prop, axis=1).reshape(-1, 1)
    '''
    #refine the prop that can prop*cell_number < 1
    
    #cell_prop[cell_prop < 1/n] = 0
    #cell_prop = cell_prop / np.sum(cell_prop, axis=1).reshape(-1, 1)
    
    #genration of expression data and fraction data

    sample = np.zeros((cell_prop.shape[0],sc_data.shape[0]))
    allcellname = meta_index.keys()
    cell_num = np.floor(n * cell_prop)
    for i in range(cell_num.shape[0]):
        if np.all(cell_num[i] == 0):
            max_idx = np.argmax(cell_prop[i])
            cell_num[i, max_idx] = n
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
    
def _get_prop_sample_bulk_specificity_bk(sc_data, meta_data, cell_composition, n_celltype, cell_specific, annotation_key, n_sample, n, round_th, project,high_purity, set_missing=False):
    """
    Get stimulated expression data and cell type prop.
        
    Parameters
    ----------
    sc_data : anndata.AnnData
        An :class:`~anndata.AnnData` containing the expression to stimulating bulk expression.
    meta_data : pandas.DataFrame
        Metadata containing cell type annotations.
    cell_composition : list
        List of cell type indices to include in simulation.
    n_celltype : int
        The cell type number used in stimulation.
    cell_specific : str
        The specific cell type that should dominate (>50% fraction).
    annotation_key : string
        String to save sc cell type information.
    n_sample : int
        The numbers of samples will be stimulated.
    n : int
        The number of cells included in each sample.
    round_th: int.
        The number to indicated the order of this round.
    project : str
        Project name for sample naming.
    
    Returns
    -------
    Returns the stimulated bulk adata.
    """
    import math
    import random
    from numpy.random import choice
    print("specificity!!!")
    meta_index = meta_data[[annotation_key]]
    meta_index = meta_index.groupby(meta_data[annotation_key]).groups
    allcellname = list(meta_index.keys())

    # Ensure cell_specific is in available cell types
    if cell_specific not in allcellname:
        raise ValueError(f"cell_specific '{cell_specific}' not found in available cell types: {allcellname}")
    
    selected_index = allcellname.index(cell_specific)
    all_cell_num = len(meta_index[cell_specific])
    
    # Adjust sample number based on the number of specific cell type
    if all_cell_num >= 1000:
        n_sample = int(n_sample * 2.5)
    elif all_cell_num >= 500:
        n_sample = int(n_sample * 2)
        
    if all_cell_num <= 30:
        n_sample = math.ceil(n_sample / 2)

    cell_prop = np.zeros((n_sample, n_celltype))
    if high_purity:
        min_val = 0.85
        max_val = 1.0
    else:
        min_val = 0.5
        max_val = 0.95
    # Generate proportions for each sample
    for i in range(n_sample):
    
        # Assign 50%-80% proportion to the specific cell type
        #specific_prop = np.random.uniform(0.5, 0.95) #sepcifity
        if high_purity:
          specific_prop = np.random.uniform(min_val, max_val) #TCGA
          cell_prop[i, selected_index] = specific_prop
          
          # Distribute remaining proportion to other cell types
          remaining_prop = 1.0 - specific_prop
          other_indices = [idx for idx in cell_composition if idx != selected_index]
        
        if len(other_indices) > 0:
            # Use Dirichlet distribution to allocate remaining proportion to other cell types
            other_weights = np.random.dirichlet(np.ones(len(other_indices)))
            other_props = other_weights * remaining_prop
            
            for j, idx in enumerate(other_indices):
                cell_prop[i, idx] = other_props[j]

    # Ensure proportions are normalized
    cell_prop = cell_prop / np.sum(cell_prop, axis=1).reshape(-1, 1)
    
    # Verify that the specific cell type proportion is above 50%
    specific_props = cell_prop[:, selected_index]
    min_specific_prop = np.min(specific_props)
    print(f"Proportion range for specific cell type '{cell_specific}': {min_specific_prop:.3f} - {np.max(specific_props):.3f}")
    
    if min_specific_prop < 0.5:
        print(f"Warning: {np.sum(specific_props < 0.5)} samples have specific cell type proportion below 50%")

    # Convert meta_index to numpy arrays
    for key, value in meta_index.items():
        meta_index[key] = np.array(value)
    
    # Generate sample expression data
    sample = np.zeros((cell_prop.shape[0], sc_data.shape[0]))
    cell_num = np.floor(n * cell_prop)
    cell_prop_new = cell_num / np.sum(cell_num, axis=1).reshape(-1, 1)
    
    for i, sample_prop in enumerate(cell_num):
        for j, cellname in enumerate(allcellname):
            if int(sample_prop[j]) > 0:  # Only sample when cell number is greater than 0
                select_index = choice(meta_index[cellname], size=int(sample_prop[j]), replace=True)
                sample[i] += sc_data.loc[:, select_index].sum(axis=1)
    
    # Normalize expression values
    sample = sample / n

    # Generate result DataFrames
    cell_prop_df = pd.DataFrame(
        cell_prop_new,
        index=[f'Sample{str(n_sample * round_th + i)}_{project}' for i in range(n_sample)],
        columns=allcellname
    )
    
    sample_data = pd.DataFrame(
        sample,
        index=[f'Sample{str(n_sample * round_th + i)}_{project}' for i in range(n_sample)],
        columns=sc_data.index
    )

    # Validate final results
    final_specific_props = cell_prop_df[cell_specific]


    return sample_data, cell_prop_df

def _get_prop_sample_bulk_specificity(sc_data, meta_data, cell_composition, n_celltype, cell_specific, annotation_key, n_sample, n, round_th, project, high_purity, celltype_prop,set_missing=False):
    """
    Get stimulated expression data and cell type prop.
        
    Parameters
    ----------
    sc_data : anndata.AnnData
        An :class:`~anndata.AnnData` containing the expression to stimulating bulk expression.
    meta_data : pandas.DataFrame
        Metadata containing cell type annotations.
    cell_composition : list
        List of cell type indices to include in simulation.
    n_celltype : int
        The cell type number used in stimulation.
    cell_specific : str
        The specific cell type that should dominate (>50% fraction).
    annotation_key : string
        String to save sc cell type information.
    n_sample : int
        The numbers of samples will be stimulated.
    n : int
        The number of cells included in each sample.
    round_th: int.
        The number to indicated the order of this round.
    project : str
        Project name for sample naming.
    high_purity : bool
        Whether to generate high purity samples.
    
    Returns
    -------
    Returns the stimulated bulk adata.
    """
    import math
    import random
    import numpy as np
    import pandas as pd
    from numpy.random import choice

    meta_index = meta_data[[annotation_key]]
    meta_index = meta_index.groupby(meta_data[annotation_key]).groups
    allcellname = list(meta_index.keys())

    # Ensure cell_specific is in available cell types
    if cell_specific not in allcellname:
        raise ValueError(f"cell_specific '{cell_specific}' not found in available cell types: {allcellname}")
    
    selected_index = allcellname.index(cell_specific)
    all_cell_num = len(meta_index[cell_specific])
    
    # Adjust sample number based on the number of specific cell type
    

    # Determine total number of samples to generate
    if high_purity and 0.75<celltype_prop:
        total_samples = n_sample * 2  # Generate both high purity and medium purity samples #2
    elif high_purity and 0.6<celltype_prop<0.75:
        total_samples = n_sample * 3  # Generate both high purity and medium purity samples #2
    else:
        total_samples = int(n_sample * 1)
        
    cell_prop = np.zeros((total_samples, n_celltype))
    
    # Generate proportions for each sample
    for i in range(total_samples):
        if high_purity and total_samples == n_sample * 3:
            if i < n_sample:
                # First n_sample: high purity (0.8-1.0)
                min_val = 0.4 #0.85 #0.5
                max_val = 1.0
            elif n_sample <=i < n_sample * 2:
                # Second n_sample: high purity (0.8-1.0)
                min_val = 0.7 #0.85 #0.7 #0.6
                max_val = 1.0
            else:
                # Next n_sample: medium purity (0.6-1.0)
                min_val = 0.6 #0.65 #0.8 #0.7
                max_val = 1.0
        elif high_purity and total_samples == n_sample * 2:
            if i < n_sample:
                # First n_sample: high purity (0.8-1.0)
                min_val = 0.4 #0.85
                max_val = 1.0
            elif n_sample <=i < n_sample * 2:
                # Second n_sample: high purity (0.8-1.0)
                min_val = 0.65 #0.85 #0.7
                max_val = 1.0
    
        else:
            # Standard purity (0.5-0.95) ori 0.6--0.95
            if n_celltype >=25:
                min_val = 0.25 #0.5 #0.3 #0.35
            elif n_celltype >=15:
                min_val = 0.4 #0.5
            else:
                min_val = 0.5
            max_val = 0.95 #1
        
        # Assign proportion to the specific cell type
        specific_prop = np.random.uniform(min_val, max_val)
        cell_prop[i, selected_index] = specific_prop
        
        # Distribute remaining proportion to other cell types
        remaining_prop = 1.0 - specific_prop
        other_indices = [idx for idx in cell_composition if idx != selected_index]
        
        if len(other_indices) > 0:
            # Use Dirichlet distribution to allocate remaining proportion to other cell types
            other_weights = np.random.dirichlet(np.ones(len(other_indices)))
            other_props = other_weights * remaining_prop
            
            for j, idx in enumerate(other_indices):
                cell_prop[i, idx] = other_props[j]

    # Ensure proportions are normalized
    #cell_prop[cell_prop < 0.02] = 0
    cell_prop = cell_prop / np.sum(cell_prop, axis=1).reshape(-1, 1)
    
    # Verify that the specific cell type proportion statistics
    specific_props = cell_prop[:, selected_index]
    min_specific_prop = np.min(specific_props)

    
    if high_purity and celltype_prop>0.6:
        high_purity_props = specific_props[:n_sample]
        medium_purity_props = specific_props[n_sample:]
       
    elif high_purity and celltype_prop<=0.6:
        high_purity_props = specific_props[:n_sample]


    # Convert meta_index to numpy arrays
    for key, value in meta_index.items():
        meta_index[key] = np.array(value)
    
    # Generate sample expression data
    sample = np.zeros((cell_prop.shape[0], sc_data.shape[0]))
    cell_num = np.floor(n * cell_prop)
    cell_prop_new = cell_num / np.sum(cell_num, axis=1).reshape(-1, 1)
    
    for i, sample_prop in enumerate(cell_num):
        for j, cellname in enumerate(allcellname):
            if int(sample_prop[j]) > 0:  # Only sample when cell number is greater than 0
                select_index = choice(meta_index[cellname], size=int(sample_prop[j]), replace=True)
                sample[i] += sc_data.loc[:, select_index].sum(axis=1)
    
    # Normalize expression values
    sample = sample / n

    # Generate result DataFrames
    cell_prop_df = pd.DataFrame(
        cell_prop_new,
        index=[f'Sample{str(total_samples * round_th + i)}_{project}' for i in range(total_samples)],
        columns=allcellname
    )
    
    sample_data = pd.DataFrame(
        sample,
        index=[f'Sample{str(total_samples * round_th + i)}_{project}' for i in range(total_samples)],
        columns=sc_data.index
    )

    # Validate final results
    final_specific_props = cell_prop_df[cell_specific]

    
    if high_purity:
        high_purity_final = final_specific_props[:n_sample]
        medium_purity_final = final_specific_props[n_sample:]
        

    return sample_data, cell_prop_df
    

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
      n_sample = n_sample
    if all_cell_num<=30:
      n_sample = math.ceil(n_sample/2)

    cell_prop = np.zeros((n_sample,n_celltype))

    get_prop = np.random.dirichlet(np.ones(len(cell_composition)), n_sample)
    for i in range(n_sample):
        cell_prop[i,cell_composition]=get_prop[i,:]
      

    cell_prop[cell_prop < 0.01] = 0


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
    '''
    for i in range(int(cell_prop.shape[1])):
        indices = np.random.choice(np.arange(cell_prop.shape[0]), replace=False, size=int(cell_prop.shape[0] * 0.1))
        cell_prop[indices, i] = 0
    '''
    cell_prop = cell_prop / np.sum(cell_prop, axis=1).reshape(-1, 1)
    for i in range(int(cell_prop.shape[0])):
        cell_prop[i,selected_index]=cell_prop[i,selected_index]+random.uniform(0.5, 2)
    #random.uniform(0, 1)
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





def pearson_similarity_matrix_vectorized(X, Y):
    """
    X: (n_samples_X, n_features)
    Y: (n_samples_Y, n_features)
    """

    X_centered = X - X.mean(axis=1, keepdims=True)
    Y_centered = Y - Y.mean(axis=1, keepdims=True)
    

    X_std = X.std(axis=1, keepdims=True)
    Y_std = Y.std(axis=1, keepdims=True)
    

    X_std = np.where(X_std == 0, 1, X_std)
    Y_std = np.where(Y_std == 0, 1, Y_std)
    

    X_normalized = X_centered / X_std
    Y_normalized = Y_centered / Y_std

    correlation_matrix = np.dot(X_normalized, Y_normalized.T) / X.shape[1]
    
    return correlation_matrix
    
    


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
                    return_adata=True,
                    specificity = False,
                    high_purity=False):
    
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

    if n_sample_each_group < 3000:
        n_sample_each_group = 3000
    n_celltype = len(cell_list)
    # Subset single-cell data
    sub_sc_adata = sc_adata[sc_adata.obs[annotation_key].isin(cell_list),:].copy()

    reference_out_dir = check_paths(out_dir+'/reference_bulk_data')
    #sub_sc_adata.write_h5ad(reference_out_dir+f"sub_sc.h5ad")
    #bulk_adata.write_h5ad(reference_out_dir+f"sub_bulk.h5ad")
    # Generate new data
    new_data = []
    new_prop = []
    if isspmatrix(sub_sc_adata.X):
          sub_sc_adata.X = sub_sc_adata.X.todense()
    from collections import Counter
    average_cell_exp = compute_cluster_averages(sub_sc_adata,annotation_key,cell_list,out_dir=out_dir,project=project,save=False).T
    if not isinstance(bulk_adata.X, np.ndarray):
        bulk_data = pd.DataFrame(bulk_adata.X.todense(),index=bulk_adata.obs_names,columns=bulk_adata.var_names)
    else:
        bulk_data = pd.DataFrame(bulk_adata.X,index=bulk_adata.obs_names,columns=bulk_adata.var_names)
    
    # Calculate sample size adjustment factor
    n_bulk_samples = bulk_data.shape[0]
    if n_bulk_samples < 100:
        if n_celltype < 15:
            sample_adjustment_factor = round(100 / n_bulk_samples)
        else:
            sample_adjustment_factor = max(round(100 / n_bulk_samples),1)
    else:
        sample_adjustment_factor = 1
    
    similarity_matrix = pearson_similarity_matrix_vectorized(bulk_data.values,average_cell_exp.values)
    all_most_index=np.argsort(-similarity_matrix, axis=1)[:, 0].flatten()
    most_sim_index = np.unique(all_most_index)
    most_sim_cell = [average_cell_exp.index.tolist()[x] for i, x in enumerate(all_most_index)]
    cell_prop = Counter(most_sim_cell)
    cell_type_counts = Counter(most_sim_cell)
    

    for cell_type, count in cell_type_counts.most_common():
        percentage = (count / n_bulk_samples) * 100

    for cell_name in cell_prop.keys():
        cell_prop[cell_name] = cell_prop[cell_name]/n_bulk_samples
    selected_index = np.argsort(-similarity_matrix, axis=1)
    selected_sim_index = np.unique(np.argsort(-similarity_matrix, axis=1).flatten())
    rare_cell_list = np.setdiff1d(selected_sim_index,most_sim_index,False)
    all_cells_names = np.array(average_cell_exp.index.tolist())[selected_sim_index]
    sample_cell_composition =  dict(enumerate(selected_index))
    
    # Calculate cell types that need specificity simulation
    # For samples with >10 cell types, identify cell types for targeted specificity simulation
    specificity_cell_types = set()
    if n_celltype > 15 and specificity:

        
        # Get top 3 most similar cell types for each sample
        top3_indices = np.argsort(-similarity_matrix, axis=1)[:, :3]
        all_top3_cells = set()
        for sample_top3 in top3_indices:
            for idx in sample_top3:
                cell_type = average_cell_exp.index.tolist()[idx]
                all_top3_cells.add(cell_type)
        
        # Get the most similar cell type for each sample
        most_sim_cells_set = set(most_sim_cell)
        
        # Calculate difference set: top3 cell types - most similar cell types
        # These cell types need additional specificity simulation
        specificity_cell_types = all_top3_cells - most_sim_cells_set
        '''
        specificity_cell_types = set(specificity_cell_types)
        #for reproduce
        EXPECTED_ORDER = [
            'NK cells', 'Macrophage', 'Luminal Progenitors', 
            'Endothelial Lymphatic LYVE1', 'PVL Differentiated', 'T cells CD4+', 
            'Endothelial ACKR1', 'Monocyte', 'Cancer Her2 SC', 
            'B cells Memory', 'CAFs myCAF-like', 'Cycling PVL', 'Cancer Cycling'
        ]
        if set(specificity_cell_types) == set(EXPECTED_ORDER):
            specificity_cell_types = [ct for ct in EXPECTED_ORDER if ct in specificity_cell_types]
        else:
        '''
        specificity_cell_types = sorted(list(specificity_cell_types))

    
    sc_data = pd.DataFrame(sub_sc_adata.X,index=sub_sc_adata.obs_names,columns=sub_sc_adata.var_names).transpose()
    
    for i in range(group_number):
        for j in range(n_bulk_samples):
            selected_celltype = sample_cell_composition[j]
            cells = np.array(average_cell_exp.index.tolist())[selected_celltype[0]]
            change_fold = cell_prop[cells]/(1/len(all_cells_names))

            if change_fold>=1.5:
                sti_num=round(n_sample_each_group*((1/change_fold)/n_bulk_samples))
                if sti_num<5:
                    sti_num=5
            elif change_fold<=0.05:
                sti_num=round(n_sample_each_group*(((1/10)/change_fold)/n_bulk_samples))
            elif change_fold<=0.1:
                sti_num=round(n_sample_each_group*(((1/4)/change_fold)/n_bulk_samples))
            elif change_fold<=0.5:
                sti_num=round(n_sample_each_group*(((1/2)/change_fold)/n_bulk_samples))
            elif change_fold<=1:
                sti_num=round(n_sample_each_group*((1/change_fold)/n_bulk_samples))
            else:
                sti_num=round(n_sample_each_group*(1/n_bulk_samples))

            # Apply sample size adjustment factor
            
            if change_fold >= 1.5 and sample_adjustment_factor>1:
                sti_num = int(sti_num * sample_adjustment_factor)
            elif change_fold >= 1 and sample_adjustment_factor>1:
                sti_num = int(sti_num * sample_adjustment_factor/4)
            elif sample_adjustment_factor>1:
                sti_num = int(sti_num * sample_adjustment_factor/8)

            
            if specificity:
                ref_data,ref_prop = _get_prop_sample_bulk_specificity(sc_data,
                                                                sub_sc_adata.obs,
                                                                selected_celltype,
                                                                n_celltype,
                                                                cells,
                                                                annotation_key,
                                                                sti_num,
                                                                min_cells_each_group+i*cell_gap_each_group,
                                                                i,
                                                                project,
                                                                high_purity,
                                                                cell_prop[cells],
                                                                set_missing=False)
                new_data.append(ref_data)
                new_prop.append(ref_prop)
            '''
            else:
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
        # Generate additional specificity samples for identified cell types
        # Sample number = max(total_samples * 0.025, 100)
        if specificity_cell_types:
            total_samples = n_sample_each_group
            specificity_sample_num = max(int(total_samples * 0.05), 100)

            specificity_sample_num = int(specificity_sample_num)
            

            
            for spec_cell_type in specificity_cell_types:

                
                # Get the index of this cell type
                spec_cell_idx = average_cell_exp.index.tolist().index(spec_cell_type)
                spec_selected_celltype = np.array([spec_cell_idx] + list(range(len(average_cell_exp))))
                #spec_selected_celltype = np.array(list(range(len(average_cell_exp))))
                # Use specificity simulation function for this cell type
                ref_data, ref_prop = _get_prop_sample_bulk_specificity(
                    sc_data,
                    sub_sc_adata.obs,
                    spec_selected_celltype,
                    n_celltype,
                    spec_cell_type,
                    annotation_key,
                    specificity_sample_num,
                    min_cells_each_group + i * cell_gap_each_group,
                    i,
                    project,
                    high_purity,
                    cell_prop[spec_cell_type],
                    set_missing=False
                )
                new_data.append(ref_data)
                new_prop.append(ref_prop)
        if high_purity and max(cell_prop.values()) > 0.85:
            random_proportion = 0.1
        elif high_purity:
            random_proportion = 0
        elif specificity:
            random_proportion = 0.5
        else:
            random_proportion = 1

        if specificity:

            ref_data,ref_prop = _get_stimulation(sc_data,
                            sub_sc_adata.obs,
                            n_celltype,
                            annotation_key,
                            int(n_sample_each_group*random_proportion),
                            min_cells_each_group+i*cell_gap_each_group,
                            i,
                            project,
                            set_missing=True)
        else:

            ref_data,ref_prop = _get_stimulation(sc_data,
                            sub_sc_adata.obs,
                            n_celltype,
                            annotation_key,
                            int(n_sample_each_group*random_proportion),
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
        # Check output path
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
                  different_source=True,
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
    if n_sample_each_group < 3000:
        n_sample_each_group = 3000
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

        start_t = time.perf_counter()

        n_celltype = len(cell_list)
        # subset sc data
        sub_sc_adata = sc_adata[sc_adata.obs[annotation_key].isin(cell_list), :].copy()
        # generate new data
        if isspmatrix(sub_sc_adata.X):
            sub_sc_adata.X = sub_sc_adata.X.todense()
        if not isinstance(sub_sc_adata.X, np.ndarray):
            sub_sc_adata.X = sub_sc_adata.X.toarray()
        new_data = []
        new_prop = []
        sc_data = pd.DataFrame(
            sub_sc_adata.X,
            index=sub_sc_adata.obs_names,
            columns=sub_sc_adata.var_names
        ).transpose()
        
        if not average_ref:
            for i in range(group_number):
                ref_data, ref_prop = _get_stimulation(
                    sc_data,
                    sub_sc_adata.obs,
                    n_celltype,
                    annotation_key,
                    n_sample_each_group,
                    min_cells_each_group + i * cell_gap_each_group,
                    i,
                    project,
                    set_missing=False
                )
                new_data.append(ref_data)
                new_prop.append(ref_prop)

            ref_data = pd.concat(new_data)
            ref_prop = pd.concat(new_prop)
        else:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # 计算每种细胞类型的平均表达
            average_cell_exp = compute_cluster_averages(
                sub_sc_adata,
                annotation_key,
                cell_list,
                out_dir=out_dir,
                project=project,
                save=True
            ).T

            if not isinstance(st_adata.X, np.ndarray):
                st_data = pd.DataFrame(
                    st_adata.X.todense(),
                    index=st_adata.obs_names,
                    columns=st_adata.var_names
                )
            else:
                st_data = pd.DataFrame(
                    st_adata.X,
                    index=st_adata.obs_names,
                    columns=st_adata.var_names
                )
            

            threshold = 1e-10
            sparsity = np.sum(np.abs(average_cell_exp.values) < threshold) / average_cell_exp.size
            similarity_matrix = pearson_similarity_matrix_vectorized(
                st_data.values,
                average_cell_exp.values
            )
            
     
            celltype_counts = sub_sc_adata.obs[annotation_key].value_counts()
            celltype_counts = celltype_counts.reindex(average_cell_exp.index, fill_value=0)

            def get_confidence_weight(n_cells, threshold=50):
                """
                Confidence weight with sharp drop below threshold
                - n_cells >= threshold: weight = 1.0
                - n_cells < threshold: weight = (n_cells/threshold)^2 (quadratic decay)
                """
                if n_cells >= threshold:
                    return 1.0
                else:
                    return (n_cells / threshold) ** 2

            confidence_weights = celltype_counts.apply(
                lambda x: get_confidence_weight(x, threshold=40)
            ).values
            

            top1_indices_before = np.argsort(-similarity_matrix, axis=1)[:, 0]
            top3_indices_before = np.argsort(-similarity_matrix, axis=1)[:, :3]
            
            all_cell_types = average_cell_exp.index.tolist()
            n_spots = st_data.shape[0]


            top1_counts_before = Counter()
            for idx in top1_indices_before:
                cell_type = all_cell_types[idx]
                top1_counts_before[cell_type] += 1


            top1_freq_before_dict = {
                ct: top1_counts_before.get(ct, 0) / n_spots
                for ct in all_cell_types
            }

            similarity_matrix = similarity_matrix * confidence_weights[np.newaxis, :]
            

            top1_indices_after = np.argsort(-similarity_matrix, axis=1)[:, 0]
            top3_indices_after = np.argsort(-similarity_matrix, axis=1)[:, :3]
            
            all_top1_cells_after = set()
            for idx in top1_indices_after:
                all_top1_cells_after.add(all_cell_types[idx])
            
            all_top3_cells_after = set()
            for spot_top3 in top3_indices_after:
                for idx in spot_top3:
                    all_top3_cells_after.add(all_cell_types[idx])
            

            top1_counts = Counter()
            top3_counts = Counter()
            
            for idx in top1_indices_after:
                cell_type = all_cell_types[idx]
                top1_counts[cell_type] += 1
            
            for spot_top3 in top3_indices_after:
                for idx in spot_top3:
                    cell_type = all_cell_types[idx]
                    top3_counts[cell_type] += 1
            

            def is_dominant(ct: str, threshold: float) -> bool:
                """Check if a cell type is dominant at given threshold (before & after weighting)."""
                top1_before = top1_freq_before_dict.get(ct, 0.0)
                top1_after = top1_counts.get(ct, 0) / n_spots
                top3_after = top3_counts.get(ct, 0) / n_spots
                return (top1_before > threshold and 
                        top1_after > threshold and 
                        top3_after > threshold)
            
            boosted_celltypes = {}
            cap_levels = [
                (0.8, 0.8),
                (0.7, 0.7),
                (0.6, 0.6),
                (0.5, 0.5)
            ]
            
            for cell_type in all_cell_types:
                top1_freq_before = top1_freq_before_dict.get(cell_type, 0.0)
                top1_freq_after = top1_counts.get(cell_type, 0) / n_spots
                top3_freq_after = top3_counts.get(cell_type, 0) / n_spots
                

                for threshold, cap_ratio in cap_levels:
                    if is_dominant(cell_type, threshold):
                        boosted_celltypes[cell_type] = cap_ratio
                        '''
                        print(
                            f"[st_simulation] Identified boosted cell type: {cell_type} "
                            f"(consistently >{threshold}: before={top1_freq_before:.3f}, "
                            f"after={top1_freq_after:.3f}, top3={top3_freq_after:.3f}) "
                            f"with boost ratio {cap_ratio}."
                        )
                        '''
                        break  # Apply only the highest applicable cap
            
            #if boosted_celltypes:
                #print(f"\n[Boosted cell types to be mixed into other samples]: {list(boosted_celltypes.keys())}")
            # ========== 识别结束 ==========

            cell_sample_allocation = allocate_samples_by_frequency(
                top1_counts=top1_counts,
                top3_counts=top3_counts,
                cell_types=all_cell_types,
                n_sample_each_group=n_sample_each_group,
                n_spots=n_spots,
                celltype_counts=celltype_counts,
                top1_freq_before_dict=top1_freq_before_dict
            )
            

            #print("\n=== SAMPLE ALLOCATION SUMMARY ===")
            #n_cell_types = len(all_cell_types)
            
            tier_stats = {'rare': [], 'top3_only': [], 'top1': []}
            
            for cell_type in all_cell_types:
                top1_freq_after = top1_counts.get(cell_type, 0) / n_spots
                top3_freq_after = top3_counts.get(cell_type, 0) / n_spots
                n_samples = cell_sample_allocation[cell_type]
                

                if top3_freq_after == 0:
                    tier = 'rare'
                    tier_stats['rare'].append(cell_type)
                elif top1_freq_after == 0:
                    tier = 'top3_only'
                    tier_stats['top3_only'].append(cell_type)
                else:
                    tier = 'top1'
                    tier_stats['top1'].append(cell_type)
                '''
                print(f"\n  {cell_type} [{tier}]:")
                print(f"    Top1 frequency (after): {top1_freq_after:.4f} ({top1_counts.get(cell_type, 0)} spots)")
                print(f"    Top3 frequency (after): {top3_freq_after:.4f} ({top3_counts.get(cell_type, 0)} spots)")
                print(f"    Allocated samples: {n_samples}")
                '''
            '''
            print(f"\n=== TIER SUMMARY ===")
            print(f"Rare (not in top3): {len(tier_stats['rare'])} types")
            if tier_stats['rare']:
                print(f"  {', '.join(tier_stats['rare'])}")
            print(f"Top3 only (not in top1): {len(tier_stats['top3_only'])} types")
            if tier_stats['top3_only']:
                print(f"  {', '.join(tier_stats['top3_only'])}")
            print(f"Top1: {len(tier_stats['top1'])} types")
            if tier_stats['top1']:
                print(f"  {', '.join(tier_stats['top1'])}")
            '''
            frequency_based_total = sum(cell_sample_allocation.values())
            #print(f"\nTotal frequency-based samples per group: {frequency_based_total}")
            
            
            random_samples_total = int(n_sample_each_group * 0.05)
            #print(f"Random samples per group: {random_samples_total}")
            

            selected_sim_index = np.arange(len(all_cell_types))
            

            for i in range(group_number):
                #print(f"\n=== Generating Group {i+1}/{group_number} ===")
                current_n_cells = min_cells_each_group + i * cell_gap_each_group
                

                #print(f"\n[Frequency-based samples]")
                for cell_type in all_cell_types:
                    n_samples = cell_sample_allocation[cell_type]
                    
                    if n_samples > 0:
                        #print(f"  Generating {n_samples} samples for {cell_type}")
                        
                        ref_data, ref_prop = _get_celltype_specific_samples(
                            sc_data,
                            sub_sc_adata.obs,
                            selected_sim_index,
                            n_celltype,
                            cell_type,
                            annotation_key,
                            n_samples,
                            current_n_cells,
                            i,
                            project,
                            sample_type='freq',
                            set_missing=False,
                            boosted_celltypes=boosted_celltypes  # 传入boosted类型
                        )
                        new_data.append(ref_data)
                        new_prop.append(ref_prop)
                

                if random_samples_total > 0:
                    #print(f"\n[Random samples]")
                    #print(f"  Generating {random_samples_total} random samples")
                    ref_data_random, ref_prop_random = _get_random_samples(
                        sc_data,
                        sub_sc_adata.obs,
                        n_celltype,
                        all_cell_types,
                        annotation_key,
                        random_samples_total,
                        current_n_cells,
                        i,
                        project,
                        set_missing=False
                    )
                    new_data.append(ref_data_random)
                    new_prop.append(ref_prop_random)
            

            ref_data = pd.concat(new_data)
            ref_prop = pd.concat(new_prop)
            

            ref_data = pd.DataFrame(
                ref_data.values,
                index=[f'Sample{str(i)}_{project}' for i in range(ref_data.shape[0])],
                columns=ref_data.columns
            )
            ref_prop = pd.DataFrame(
                ref_prop.values,
                index=[f'Sample{str(i)}_{project}' for i in range(ref_data.shape[0])],
                columns=ref_prop.columns
            )
            
        if rename_dict is not None:
            ref_prop.rename(columns=rename_dict, inplace=True) 
            
        #print(f'\nTime to generate st data: {round(time.perf_counter() - start_t, 2)} seconds')
        #print(f'Total samples generated: {ref_data.shape[0]}')


        print('Finish st stimulation.')
        print('================================================================================================================')

    if not return_adata:
        if save:
            reference_out_dir = check_paths(out_dir+'/reference_st_data')
            print('Saving stimulated data')
            ref_data.to_csv(reference_out_dir+f"/{project}_stimulated_st_exp.txt", sep='\t')
            ref_prop.to_csv(reference_out_dir+f"/{project}_stimulated_st_prop.txt", sep='\t')
        return ref_data, ref_prop
    else:
        adata = sc.AnnData(ref_data)
        adata.obs = ref_prop
        if save:
            reference_out_dir = check_paths(out_dir+'/reference_st_data')
            print('Saving stimulated data')
            ref_data.to_csv(reference_out_dir+f"/{project}_stimulated_st_exp.txt", sep='\t')
            ref_prop.to_csv(reference_out_dir+f"/{project}_stimulated_st_prop.txt", sep='\t')
            adata.write_h5ad(f"{reference_out_dir}/filtered_{project}_st.h5ad") 

        return adata
        


def allocate_samples_by_frequency(
    top1_counts: dict,
    top3_counts: dict,
    cell_types: list,
    n_sample_each_group: int,
    n_spots: int,
    celltype_counts: pd.Series,
    top1_freq_before_dict: dict,  # top1 frequency before weighting
) -> dict:
    """
    Allocate sample numbers based on top1 and top3 frequencies.

    Key rules:
    1. If the cell type with the largest single-cell count has ≥3× cells compared 
       to the second largest, boost its allocation.
    2. Only trigger boost/cap rules when BOTH pre-weight and post-weight frequencies 
       meet the threshold (consistently high).
    3. If dominant type detected, double the scale_factor to increase diversity.
    """

    n_cell_types = len(cell_types)
    cell_sample_allocation = {}

    # -------- 1. Find global top1/top2 by single-cell counts --------
    counts_for_alloc = (
        celltype_counts
        .reindex(cell_types)
        .fillna(0)
        .astype(int)
    )

    sorted_by_count = counts_for_alloc.sort_values(ascending=False)

    if len(sorted_by_count) > 0:
        global_top1_type = sorted_by_count.index[0]
        global_top1_count = int(sorted_by_count.iloc[0])
    else:
        global_top1_type, global_top1_count = None, 0

    if len(sorted_by_count) > 1:
        global_top2_count = int(sorted_by_count.iloc[1])
    else:
        global_top2_count = 0

    very_large_pool = False
    if global_top1_type is not None and global_top2_count > 0 and n_cell_types > 10:
        if global_top1_count >= 3 * global_top2_count:
            very_large_pool = True
            '''
            print(
                f"[allocate_samples_by_frequency] Very large pool detected: "
                f"{global_top1_type} (cells={global_top1_count}, "
                f"second={global_top2_count}, ratio={global_top1_count/global_top2_count:.2f} ≥ 3)."
            )
            '''
    '''
    else:
        print("[allocate_samples_by_frequency] Cannot compute 'very_large_pool' "
              "(maybe only 1 cell type or second has 0 cells).")
    '''

    # -------- 1.5. Detect dominant types (consistently high in both pre & post weighting) --------
    def is_dominant(ct: str, threshold: float) -> bool:
        """Check if a cell type is dominant at given threshold (before & after weighting)."""
        top1_before = top1_freq_before_dict.get(ct, 0.0)
        top1_after = top1_counts.get(ct, 0) / n_spots
        top3_after = top3_counts.get(ct, 0) / n_spots
        return (top1_before > threshold and 
                top1_after > threshold and 
                top3_after > threshold)

    has_dominant_type = any(is_dominant(ct, 0.5) for ct in cell_types)
    
    if has_dominant_type:
        dominant_types = [ct for ct in cell_types if is_dominant(ct, 0.5)]
        

    # -------- 2. Determine scale_factor --------
    base_scale_factor = 2 if len(cell_types) > 15 else 1
    scale_factor = base_scale_factor * 2 if has_dominant_type else base_scale_factor
    
        
    # -------- 3. Determine max_samples --------
    if len(cell_types) < 15:
        max_samples = int(n_sample_each_group * 0.35)
    else:

        max_samples = int(n_sample_each_group * 0.3)

    # -------- 4. Allocate samples for each cell type --------
    for cell_type in cell_types:
        top1_freq_after = top1_counts.get(cell_type, 0) / n_spots
        top3_freq_after = top3_counts.get(cell_type, 0) / n_spots
        top1_freq_before = top1_freq_before_dict.get(cell_type, 0.0)

        # (1) Not in top3: rare type
        if top3_freq_after == 0:
            n_samples = max(int(n_sample_each_group * 0.01 * scale_factor), 1)

        # (2) In top3 but never in top1
        elif top1_freq_after == 0:
            if n_cell_types > 5:
                #divisor = 3
                divisor = 4 if top3_freq_after < 0.005 else 3
            else:
                divisor = 4
            n_samples = max(
                int((n_sample_each_group / (divisor * n_cell_types)) * scale_factor),
                1
            )
        
        # (3) Appears in top1
        else:

            min_samples = max(
                int((n_sample_each_group / (2 * n_cell_types)) * scale_factor),
                1
            )

            freq_samples = int(top1_freq_after * n_sample_each_group)
            if has_dominant_type:
                  freq_samples = int(freq_samples * scale_factor / 2)

            n_samples = max(min(freq_samples, max_samples), min_samples)

            # Boost very large pool only if consistently high frequency (before & after)
            if (very_large_pool and 
                cell_type == global_top1_type and 
                n_samples < max_samples):
                n_samples = int(n_sample_each_group * 0.5)

        # -------- Boost for high top3 but low top1 (mixed composition) --------
        if (top3_freq_after > 0.5 and
            top1_freq_after < 0.1 and
            n_cell_types > 5):

            boosted_samples = n_samples * 2
            n_samples = min(boosted_samples, max_samples)
            
        # -------- Cap dominant types (only if consistently high) --------
        cap_levels = [
            (0.8, 0.8),
            (0.7, 0.7),
            (0.6, 0.6),
            (0.5, 0.5)
        ]
        
        for threshold, cap_ratio in cap_levels:
            if is_dominant(cell_type, threshold):
                n_samples = int(n_sample_each_group * cap_ratio)
                break  # Apply only the highest applicable cap

        cell_sample_allocation[cell_type] = n_samples

    return cell_sample_allocation
    
    

                   


    
def _get_celltype_specific_samples(sc_data, meta_data, cell_composition, n_celltype, 
                                   cell_specific, annotation_key, n_sample, n, 
                                   round_th, project, sample_type='freq', set_missing=False,
                                   boosted_celltypes=None):
    """
    Generate samples with a specific cell type enriched.
    
    Parameters
    ----------
    sc_data : pd.DataFrame
        Single-cell expression data (genes x cells).
    meta_data : pd.DataFrame
        Metadata containing cell type annotations.
    cell_composition : array
        Indices of cell types that can be included.
    n_celltype : int
        Total number of cell types.
    cell_specific : str
        The target cell type to enrich in samples.
    annotation_key : str
        Column name in meta_data for cell type annotations.
    n_sample : int
        Number of samples to generate.
    n : int
        Number of cells per sample.
    round_th : int
        Round indicator for naming.
    project : str
        Project name for sample naming.
    sample_type : str
        'freq' for frequency-based or 'random' for random samples.
    set_missing : bool
        Whether to set some cell types to zero.
    boosted_celltypes : dict, optional
        Dictionary of {celltype: max_fraction} for cell types that need to be 
        mixed into other samples.
    
    Returns
    -------
    sample_data : pd.DataFrame
        Simulated expression data.
    cell_prop : pd.DataFrame
        Cell type proportions for each sample.
    """
    
    meta_index = meta_data[[annotation_key]]
    meta_index = meta_index.groupby(meta_data[annotation_key]).groups
    allcellname = list(meta_index.keys())
    
    # Find the index of target cell type
    try:
        selected_index = allcellname.index(cell_specific)
    except ValueError:
        print(f"Warning: {cell_specific} not found in cell types, skipping.")
        return pd.DataFrame(), pd.DataFrame()
    
    # Prepare meta_index as numpy array
    for key, value in meta_index.items():
        meta_index[key] = np.array(value)
    
    # Initialize proportion matrix
    cell_prop = np.zeros((n_sample, n_celltype))
    
    # Check if current cell_specific is a boosted type
    is_current_boosted = (boosted_celltypes is not None and 
                         cell_specific in boosted_celltypes)
    
    # Different strategies based on n
    if n < 10:
        
        target_min_fraction = 0.1
        
        max_type = n
        if len(cell_composition) > max_type:
            get_prop = np.random.dirichlet(np.ones(max_type), n_sample)
            for j in range(n_sample):
                cell_chosen = np.random.choice(cell_composition, replace=False, size=max_type).tolist()
                if selected_index not in cell_chosen:
                    cell_chosen[-1] = selected_index
                cell_prop[j, cell_chosen] = get_prop[j, :]
        else:
            get_prop = np.random.dirichlet(np.ones(len(cell_composition)), n_sample)
            for i in range(n_sample):
                cell_prop[i, cell_composition] = get_prop[i, :]
        
        # Ensure target cell type fraction > 0
        for i in range(n_sample):
            if cell_prop[i, selected_index] < target_min_fraction:
                other_cells = cell_prop[i, :] > 0
                other_cells[selected_index] = False
                if np.any(other_cells):
                    reduction = target_min_fraction - cell_prop[i, selected_index]
                    total_other = np.sum(cell_prop[i, other_cells])
                    if total_other > reduction:
                        cell_prop[i, other_cells] -= (cell_prop[i, other_cells] / total_other) * reduction
                    else:
                        cell_prop[i, :] = 0
                        cell_prop[i, selected_index] = target_min_fraction
                        remaining = 1 - target_min_fraction
                        if np.sum(other_cells) > 0:
                            cell_prop[i, other_cells] = remaining / np.sum(other_cells)
                    cell_prop[i, selected_index] = target_min_fraction
    else:
        # When n>=10
        if n <= 5:
            target_cell_num = n
        else:
            target_cell_num = min(4, len(cell_composition))
        
        if len(cell_composition) > target_cell_num:
            get_prop = np.random.dirichlet(np.ones(target_cell_num), n_sample)
            for j in range(n_sample):
                cell_chosen = np.random.choice(cell_composition, replace=False, size=target_cell_num).tolist()
                if selected_index not in cell_chosen:
                    cell_chosen.pop()
                    cell_chosen.append(selected_index)
                cell_prop[j, cell_chosen] = get_prop[j, :]
        else:
            get_prop = np.random.dirichlet(np.ones(len(cell_composition)), n_sample)
            for i in range(n_sample):
                cell_prop[i, cell_composition] = get_prop[i, :]
        
        # Sample target fraction for main cell type
        for i in range(n_sample):
            # If current cell type is boosted, use different minimum
            if is_current_boosted:
                if n < 15:
                    target_fraction = np.random.uniform(0.2, 1)
                else:
                    target_fraction = np.random.uniform(0.2, 1)
            else:
                if n < 15:
                    target_fraction = np.random.uniform(0.1, 1)
                else:
                    target_fraction = np.random.uniform(0.1, 1)
            
            current_fraction = cell_prop[i, selected_index]
            
            if current_fraction < target_fraction:
                deficit = target_fraction - current_fraction
                other_cells = np.arange(n_celltype) != selected_index
                total_other = np.sum(cell_prop[i, other_cells])
                
                if total_other > deficit:
                    scale_factor = (total_other - deficit) / total_other
                    cell_prop[i, other_cells] *= scale_factor
                else:
                    cell_prop[i, other_cells] = 0
                
                cell_prop[i, selected_index] = target_fraction
                remaining = 1 - target_fraction
                if total_other > 0 and remaining > 0:
                    cell_prop[i, other_cells] = (cell_prop[i, other_cells] / (np.sum(cell_prop[i, other_cells]) + 1e-10)) * remaining
            else:
                other_cells = np.arange(n_celltype) != selected_index
                cell_prop[i, selected_index] = target_fraction
                remaining = 1 - target_fraction
                total_other = np.sum(cell_prop[i, other_cells])
                if total_other > 0 and remaining > 0:
                    cell_prop[i, other_cells] = (cell_prop[i, other_cells] / total_other) * remaining
    
    # Renormalize to ensure sum equals 1
    cell_prop = cell_prop / np.sum(cell_prop, axis=1).reshape(-1, 1)
    
    # Convert proportions to cell counts
    cell_num = np.floor(n * cell_prop)
    
    # Ensure total cell count equals n
    for i in range(cell_num.shape[0]):
        current_total = int(np.sum(cell_num[i]))
        if current_total < n:
            cell_num[i, selected_index] += (n - current_total)
        elif current_total > n:
            excess = current_total - n
            for j in range(n_celltype):
                if j != selected_index and cell_num[i, j] > 0:
                    reduction = min(excess, int(cell_num[i, j]))
                    cell_num[i, j] -= reduction
                    excess -= reduction
                    if excess == 0:
                        break
        
        if np.sum(cell_num[i]) == 0:
            cell_num[i, selected_index] = n
    
    # ========== Mix in boosted cell types (after cell_num is finalized) ==========
    if boosted_celltypes is not None and not is_current_boosted:
        for boosted_ct, max_boosted_fraction in boosted_celltypes.items():
            if boosted_ct == cell_specific:
                continue  # Skip current main type
            
            try:
                boosted_idx = allcellname.index(boosted_ct)
            except ValueError:
                continue  # Skip if boosted type doesn't exist
            
            # Determine boost_ratio based on max_boosted_fraction
            if max_boosted_fraction >= 0.8:
                boost_ratio = 0.8
            elif max_boosted_fraction >= 0.7:
                boost_ratio = 0.7
            elif max_boosted_fraction >= 0.6:
                boost_ratio = 0.6
            else:
                boost_ratio = 0.5  # Default
            boost_ratio = boost_ratio + 0.1
            #print(f"boost_ratio: {boost_ratio}")
            
            # Randomly select samples to receive boosted types
            n_samples_to_boost = int(n_sample * boost_ratio)
            samples_to_boost = np.random.choice(n_sample, size=n_samples_to_boost, replace=False)
            
            # Mix boosted type into selected samples
            for i in samples_to_boost:
                # Verify and fix total before boost
                current_total = int(np.sum(cell_num[i]))
                if current_total != n:
                    cell_num[i, selected_index] += (n - current_total)
                
                # Find non-specific cell types that have cells
                non_specific_mask = np.arange(n_celltype) != selected_index
                available_types = np.where((cell_num[i] > 0) & non_specific_mask)[0]
                
                if len(available_types) == 0:
                    continue  # No other cell types to replace, keep unchanged
                
                # Randomly select one non-specific cell type to replace
                replace_idx = np.random.choice(available_types)
                
                # Transfer all cells from selected type to boosted type
                cells_to_transfer = int(cell_num[i, replace_idx])
                cell_num[i, boosted_idx] += cells_to_transfer
                cell_num[i, replace_idx] = 0

    # ========== End of mixing logic ==========
    
    # Recalculate actual proportions
    cell_prop_new = cell_num / np.sum(cell_num, axis=1).reshape(-1, 1)
    
    # Generate simulated samples
    sample = np.zeros((cell_prop.shape[0], sc_data.shape[0]))
    for i, sample_prop in enumerate(cell_num):
        for j, cellname in enumerate(allcellname):
            num_cells = int(sample_prop[j])
            if num_cells > 0:
                select_index = choice(meta_index[cellname], size=num_cells, replace=True)
                sample[i] += sc_data.loc[:, select_index].sum(axis=1)
    
    # Normalize
    sample = sample / n
    
    # Create DataFrames
    sample_suffix = f'{sample_type}_{cell_specific}'
    cell_prop_df = pd.DataFrame(
        cell_prop_new,
        index=[f'Sample{str(i)}_{project}_{sample_suffix}' 
               for i in range(n_sample)],
        columns=allcellname
    )
    
    sample_data_df = pd.DataFrame(
        sample,
        index=[f'Sample{str(i)}_{project}_{sample_suffix}' 
               for i in range(n_sample)],
        columns=sc_data.index
    )
    
    return sample_data_df, cell_prop_df
    
    



def _get_random_samples(sc_data, meta_data, n_celltype, all_cell_types,
                       annotation_key, n_sample, n, round_th, project, 
                       set_missing=False):
    """
    Generate random samples without cell type enrichment.
    
    Parameters
    ----------
    sc_data : pd.DataFrame
        Single-cell expression data (genes x cells).
    meta_data : pd.DataFrame
        Metadata containing cell type annotations.
    n_celltype : int
        Total number of cell types.
    all_cell_types : list
        List of all cell type names.
    annotation_key : str
        Column name in meta_data for cell type annotations.
    n_sample : int
        Number of samples to generate.
    n : int
        Number of cells per sample.
    round_th : int
        Round indicator for naming.
    project : str
        Project name for sample naming.
    set_missing : bool
        Whether to set some cell types to zero.
    
    Returns
    -------
    sample_data : pd.DataFrame
        Simulated expression data.
    cell_prop : pd.DataFrame
        Cell type proportions for each sample.
    """
    
    meta_index = meta_data[[annotation_key]]
    meta_index = meta_index.groupby(meta_data[annotation_key]).groups
    allcellname = list(meta_index.keys())

    for key, value in meta_index.items():
        meta_index[key] = np.array(value)
    

    cell_prop = np.zeros((n_sample, n_celltype))
    

    for i in range(n_sample):
        n_types = np.random.randint(3, min(5, n_celltype + 1))
        selected_types = np.random.choice(range(n_celltype), size=n_types, replace=False)
        

        prop = np.random.dirichlet(np.ones(n_types))
        cell_prop[i, selected_types] = prop
    

    cell_num = np.floor(n * cell_prop)
    

    for i in range(cell_num.shape[0]):
        if np.sum(cell_num[i]) < n:

            available_types = np.where(cell_num[i] > 0)[0]
            if len(available_types) > 0:
                chosen_type = np.random.choice(available_types)
                cell_num[i, chosen_type] += (n - np.sum(cell_num[i]))
            else:

                chosen_type = np.random.randint(0, n_celltype)
                cell_num[i, chosen_type] = n
    

    cell_prop_new = cell_num / np.sum(cell_num, axis=1).reshape(-1, 1)
    

    sample = np.zeros((cell_prop.shape[0], sc_data.shape[0]))
    for i, sample_prop in enumerate(cell_num):
        for j, cellname in enumerate(allcellname):
            num_cells = int(sample_prop[j])
            if num_cells > 0:
                select_index = choice(meta_index[cellname], size=num_cells, replace=True)
                sample[i] += sc_data.loc[:, select_index].sum(axis=1)
    

    sample = sample / n
    

    cell_prop_df = pd.DataFrame(
        cell_prop_new,
        index=[f'Sample{str(i)}_{project}_random' 
               for i in range(n_sample)],
        columns=allcellname
    )
    
    sample_data_df = pd.DataFrame(
        sample,
        index=[f'Sample{str(i)}_{project}_random' 
               for i in range(n_sample)],
        columns=sc_data.index
    )
    
    return sample_data_df, cell_prop_df
    
    




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