import numpy as np
import pandas as pd
import random
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import time
from .. import utils
from ortools.graph import pywrapgraph


def downsample(data_df, target_count):
    """
    Downsamples dataset to target_count transcripts per cell.

    Parameters :
        data_df (pd.DataFrame) :
            data to downsample, with rows as genes and columns as cells.

    Returns :
        downsampled_df (pd.DataFrame) :
            downsampled data, where the sum of each column is target_count
            (or lower, for columns whose sum was originally lower than target_count).
    """
    def downsample_cell(sr, target_tr_count):
        if sr.sum() <= target_tr_count:
            return sr

        genes, counts = np.unique(np.random.choice(np.repeat(sr.index, sr.to_numpy()), target_tr_count), return_counts=True)
        downsampled = pd.Series(counts, index=genes).reindex(sr.index, fill_value=0)

        return downsampled

    downsampled_df = data_df.apply(lambda k: downsample_cell(k, target_count), axis=0)

    return downsampled_df

def estimate_cell_num(st_data, mean_cell_numbers):
    # Read data
    expressions = st_data.values.astype(float)

    # Data normalization
    expressions_tpm_log = utils.normalize_data(expressions)

    # Set up fitting problem
    RNA_reads = np.sum(expressions_tpm_log, axis=0, dtype=float)
    mean_RNA_reads = np.mean(RNA_reads)
    min_RNA_reads = np.min(RNA_reads)

    min_cell_numbers = 1 if min_RNA_reads > 0 else 0

    fit_parameters = np.polyfit(np.array([min_RNA_reads, mean_RNA_reads]),
                                np.array([min_cell_numbers, mean_cell_numbers]), 1)
    polynomial = np.poly1d(fit_parameters)
    cell_number_to_node_assignment = polynomial(RNA_reads).astype(int)


    cell_number_to_node_assignment = pd.DataFrame(cell_number_to_node_assignment,index=st_data.columns,columns=["cell_num"])


    return cell_number_to_node_assignment

def get_cell_type_fraction(number_of_cells, cell_type_fraction_data):
    # Uncomment commented lines for closer numbers
    fraction_mean = cell_type_fraction_data.mean(axis=0).to_frame()
    fractions_to_numbers = fraction_mean[0].values*number_of_cells.values
    fractions_to_numbers = fractions_to_numbers.astype(int)
    fraction_mean.iloc[:,0] = fractions_to_numbers
    #cell_type_numbers.loc[max_ct,cell_type_numbers.columns[0]] += number_of_cells - sum(cell_type_numbers.iloc[:,0])
    fraction_mean.loc[fraction_mean.index[0],fraction_mean.columns[0]] += number_of_cells.values - sum(fraction_mean.iloc[:,0])
    return fraction_mean


def modify_row(row):
    max_value = row.max()
    if max_value < 0.5:
        return np.where(row == max_value, 1, 0)
    else:
        return np.round(row)



def sample_single_cells(scRNA_data, cell_type_data, cell_type_numbers_int, sampling_method, seed):
    """
    Samples cells from scRNA_data based on the cell type distribution specified in cell_type_numbers_int.
    The sampled count for each cell type will match the number specified in cell_type_numbers_int.

    Parameters :
        scRNA_data             (2D pd.DataFrame) : gene x cell scRNA expression data to be sampled from.
        cell_type_data        (nx1 pd.DataFrame) : cell types corresponding to scRNA_data. (index=CellID)
        cell_type_numbers_int (nx1 pd.DataFrame) : ST cell count for each cell type. (index=CellType)
        sampling_method (str), seed (int) : as specified in args.
    
    Returns :
        all_cells_save         (2D pd.DataFrame) : gene x cell scRNA expression data for sampled single cells.
            This will be a subset (likely with duplicate columns) of the provided scRNA_data (sampling_method == "duplicates"),
            otherwise a superset of the provided scRNA_data with newly generated placeholder cells (sampling_method == "place_holders").
            It is guaranteed that the cells will be in order of cell types as specified in cell_type_numbers_int.index.
            index=gene, columns=CellID.
    """    
    np.random.seed(seed)
    random.seed(seed)

    # Down/up sample of scRNA-seq data according to estimated cell type fractions
    # follow the order of cell types in cell_type_numbers_int
    unique_cell_type_labels = cell_type_numbers_int.index.values
    # initialize variables
    all_cells_save_list = [] # List of 2D np.ndarray of single cell expression
    cell_names_list = [] # List of 1D np.array of single cell IDs
    sampled_index_total = [] # List of 1D np.array of single cell indices

    for cell_type in unique_cell_type_labels:
        cell_type_index = cell_type_data.index[cell_type_data.values == cell_type].tolist()
        #cell_type_index = np.nonzero(cell_type_data.values == cell_type)[0].tolist()
        #cell_type_index = cell_type_data.index.values[cell_type_index]
        cell_type_count_available = len(cell_type_index)
        if cell_type_count_available == 0:
            raise ValueError(f"Cell type {cell_type} in the ST dataset is not available in the scRNA-seq dataset.")
        cell_type_count_desired = cell_type_numbers_int.loc[cell_type,0]

        if sampling_method == "duplicates":
            if cell_type_count_desired > cell_type_count_available:
                cell_type_selected_index = np.concatenate([
                    cell_type_index, np.random.choice(cell_type_index, int(cell_type_count_desired) -int(cell_type_count_available))
                ], axis=0) # ensure at least one copy of each, then sample the rest

            else:
                cell_type_selected_index = random.sample(cell_type_index, int(cell_type_count_desired))

        
            sampled_index_total.append(cell_type_selected_index)
        
        else:
            raise ValueError("Invalid sampling_method provided")

    sampled_index_total = np.concatenate(sampled_index_total, axis=0)
    all_cells_save = scRNA_data.loc[:, sampled_index_total]

    return all_cells_save


def partition_indices(indices, split_by_category_list=None, split_by_interval_int=None, shuffle=True):
    """
    Splits the provided indices into list of smaller index sets based on other parameters.
    indices is originally a single 1D numpy array, which is then split and returned as a list of smaller 1D numpy arrays.
    e.g., indices = np.arange(0, 5000) can be split into [np.arange(0, 2000), np.arange(2000, 5000)].

    Parameters :
        indices                (1D np.array(int)) : indices to be split.

        split_by_category_list (1D np.array(int)) : number of indices for each category that cannot be mixed together.
            split_by_category_list should sum to len(indices)
            e.g., if split_by_category_list is [3000, 5000], with indices == 0:8000,
                    then the first 3000 will be partitioned separately from the latter 5000.
                    i.e., [0:2000, 2000:3000, 3000:7000, 7000:8000] is possible, but [0:2000, 2000:7000, 7000:8000] is not.
        
        split_by_interval_int               (int) : max length of each partition.
            if split_by_category_list is None, then indices are split into partitions of this size.
            e.g., if split_by_interval_int is 1000, with indices == 0:2500, then [0:1000, 1000:2000, 2000:2500] is returned.
            if split_by_category_list is specified, then any category that exceeds this size will be further partitioned.
            e.g., if split_by_category_list is [500, 1000, 300] and split_by_interval_int is 400, with indices == 0:1800,
                    then [0:400, 400:500, 500:900, 900:1300, 1300:1500, 1500:1800] is returned.
                    - split_by_category_list sets breakpoints at 500 and 1500
                    - split_by_interval_int sets breakpoints at every 400 inside each of the three groups

        shuffle                            (bool) : whether indices should be shuffled before being split.
    
    Returns :
        List[1D np.array(int)] : Partitioned indices.
    """
    num_indices = len(indices)

    if shuffle:
        np.random.shuffle(indices)

    # initialize breakpoints, as start and end
    breakpoints = [0, num_indices]

    # add breakpoints set by split_by_category_list
    if split_by_category_list is not None:
        if np.sum(split_by_category_list) != num_indices:
            print('Warning: sum of counts in each category does not match the full length')
        breakpoints.extend(np.cumsum(split_by_category_list))
        breakpoints = sorted(np.unique(breakpoints))
    
    # add breakpoints set by split_by_interval_int
    if split_by_interval_int is not None:
        breakpoints_new = breakpoints.copy()
        for idx in range(len(breakpoints)-1):
            if breakpoints[idx+1] - breakpoints[idx] <= split_by_interval_int:
                continue
            breakpoints_new.extend(np.arange(breakpoints[idx], breakpoints[idx+1], split_by_interval_int))
        breakpoints = breakpoints_new
    
    # take unique values, sort in ascending order, and remove start and end indices
    breakpoints = sorted(np.unique(breakpoints))[1:-1]

    # partition the indices based on breakpoints
    split_indices_list = np.array_split(indices, breakpoints)

    return split_indices_list




def build_cost_matrix(expressions_tpm_st_log, expressions_tpm_scRNA_log, keep_rate,cell_number_to_node_assignment):
    start_time = time.time()

    to_assign = expressions_tpm_scRNA_log.shape[1]
    if (to_assign > 100):
        to_assign = int(to_assign * keep_rate)
    print(to_assign)
    knn = NearestNeighbors(n_neighbors=to_assign)

    fit_results = knn.fit(expressions_tpm_scRNA_log.T.to_numpy())


    distances, indices = knn.kneighbors(expressions_tpm_st_log.T.to_numpy())

    d_new = np.zeros((expressions_tpm_st_log.shape[1], expressions_tpm_scRNA_log.shape[1])) # z-by-y 


    for i in range(expressions_tpm_st_log.shape[1]): 
        for idx, dist in zip(indices[i], distances[i]):
            d_new[i, idx] = dist 


    location_repeat = np.zeros(d_new.shape[1])
    counter = 0
    location_repeat = np.repeat(np.arange(len(cell_number_to_node_assignment)), cell_number_to_node_assignment)

    location_repeat = location_repeat.astype(int)
    distance_repeat = d_new[location_repeat, :]
    lap_expressions_tpm_st_log = expressions_tpm_st_log.iloc[:,location_repeat]
    matrix = csr_matrix(distance_repeat)
    end_time = time.time()
    duration = end_time - start_time
    
    return matrix,lap_expressions_tpm_st_log

def lap(matrix):
    start_time = time.time()
    rows, cols = matrix.shape
    assignment_mat = np.full(rows, -1)
    assignment = pywrapgraph.SimpleMinCostFlow()
    
    for i in range(rows):
        start = matrix.indptr[i]
        end = matrix.indptr[i + 1]
        for index in range(start, end):
            j = int(matrix.indices[index])
            cost_value = int(matrix.data[index])
            assignment.AddArcWithCapacityAndUnitCost(i, rows + j, 1, cost_value)

    for i in range(rows):
        assignment.SetNodeSupply(i, 1)
    for j in range(cols):
        assignment.SetNodeSupply(rows + j, -1)
    

    if assignment.SolveMaxFlowWithMinCost() == assignment.OPTIMAL:
        for i in range(assignment.NumArcs()):
            if assignment.Flow(i) > 0:
                assignment_mat[assignment.Tail(i)] = assignment.Head(i) - rows
    else:
        print('There was an issue with the min cost flow input.')
    
    end_time = time.time()
    duration = end_time - start_time

    
    return assignment_mat, assignment.OptimalCost()
