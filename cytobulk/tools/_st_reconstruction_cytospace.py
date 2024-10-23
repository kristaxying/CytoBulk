import numpy as np
import pandas as pd
import random
import time
import warnings
from scipy.spatial import distance


def import_solver(solver_method):
    try:
        if solver_method == "lapjv_compat":
            from lap import lapjv
            solver = lapjv
        elif solver_method == "lapjv":
            from lapjv import lapjv
            solver = lapjv
        else:
            raise NotImplementedError(f"The solver {solver_method} is not a supported solver "
                                      "for the shortest augmenting path method, choose between "
                                      "'lapjv' and 'lapjv_compat'.")
    except ModuleNotFoundError:
        raise ModuleNotFoundError("The Python package containing the solver_method option "
                                  f"you have chosen {solver_method} was not found. If you "
                                  "selected 'lapjv_compat' solver, install package 'lap'"
                                  "by running 'pip install lap==0.4.0'. If you selected 'lapjv'"
                                  "solver, install package 'lapjv' by running `pip install lapjv==1.3.14'"
                                  "or check the package home page for further instructions.")

    return solver

def normalize_data(data):
    data = np.nan_to_num(data).astype(float)
    data *= 10**6 / np.sum(data, axis=0, dtype=float)
    np.log2(data + 1, out=data)
    np.nan_to_num(data, copy=False)
    return data
       
def matrix_correlation_pearson(v1, v2):
    if v1.shape[0] != v2.shape[0]:
        raise ValueError("The two matrices v1 and v2 must have equal dimensions; ST and scRNA data must have the same genes")

    n = v1.shape[0]
    sums = np.multiply.outer(v2.sum(0), v1.sum(0))
    stds = np.multiply.outer(v2.std(0), v1.std(0))
    correlation = (v2.T.dot(v1) - sums / n) / stds / n

    return correlation


def matrix_correlation_spearman(v1, v2):
    
    if v1.shape[0] != v2.shape[0]:
        raise ValueError("The two matrices v1 and v2 must have equal dimensions; ST and scRNA data must have the same genes")
        
    v1 = pd.DataFrame(v1).rank().values
    v2 = pd.DataFrame(v2).rank().values
    
    n = v1.shape[0] # v1 and v2 should have the same number of rows
    sums = np.multiply.outer(v2.sum(0), v1.sum(0))
    stds = np.multiply.outer(v2.std(0), v1.std(0))
    correlation = (v2.T.dot(v1) - sums / n) / stds / n
    
    return correlation


def call_solver(solver, solver_method, cost_scaled):
    if solver_method == "lapjv_compat":
        _, _, y = solver(cost_scaled)
    elif solver_method == "lapjv":
        _, y, _ = solver(cost_scaled)

    return y

def calculate_cost(expressions_tpm_scRNA_log, expressions_tpm_st_log, cell_number_to_node_assignment,
                    solver_method, distance_metric):
    print("Building cost matrix ...")
    t0 = time.perf_counter()
    if solver_method=="lap_CSPR":
        if distance_metric=="Pearson_correlation":
           cost = -np.transpose(matrix_correlation_pearson(expressions_tpm_st_log, expressions_tpm_scRNA_log))
        elif distance_metric=="Spearman_correlation":
           cost = -np.transpose(matrix_correlation_spearman(expressions_tpm_st_log, expressions_tpm_scRNA_log))
        elif distance_metric=="Euclidean":
           cost = np.transpose(distance.cdist(np.transpose(expressions_tpm_scRNA_log), np.transpose(expressions_tpm_st_log), 'euclidean'))
    else:
        if distance_metric=="Pearson_correlation":
           cost = -matrix_correlation_pearson(expressions_tpm_scRNA_log, expressions_tpm_st_log)
        elif distance_metric=="Spearman_correlation":
           cost = -matrix_correlation_spearman(expressions_tpm_scRNA_log, expressions_tpm_st_log)
        elif distance_metric=="Euclidean":
           cost = np.transpose(distance.cdist(np.transpose(expressions_tpm_scRNA_log), np.transpose(expressions_tpm_st_log), 'euclidean'))

    location_repeat = np.zeros(cost.shape[1])
    counter = 0
    location_repeat = np.repeat(np.arange(len(cell_number_to_node_assignment)), cell_number_to_node_assignment)

    location_repeat = location_repeat.astype(int)
    distance_repeat = cost[location_repeat, :]
    print(f"Time to build cost matrix: {round(time.perf_counter() - t0, 2)} seconds")

    return distance_repeat, location_repeat


def match_solution(cost):
    rows = len(cost)
    cols = len(cost[0])
    assignment_mat = np.zeros((rows, 2))
    assignment = pywrapgraph.LinearSumAssignment()
    for worker in range(rows):
        for task in range(cols):
            if cost[worker][task]:
                assignment.AddArcWithCost(worker, task, cost[worker][task])

    solve_status = assignment.Solve()
    if solve_status == assignment.OPTIMAL:
        print('Total cost = ', assignment.OptimalCost())
        print()
        for i in range(0, assignment.NumNodes()):
            assignment_mat[i, 0] = assignment.RightMate(i)
            assignment_mat[i, 1] = assignment.AssignmentCost(i)
    elif solve_status == assignment.INFEASIBLE:
        print('No assignment is possible.')
    elif solve_status == assignment.POSSIBLE_OVERFLOW:
        print('Some input costs are too large and may cause an integer overflow.')
    else:
        raise ValueError("The assignment failed")
    
    return assignment_mat


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
    

def read_data(scRNA_path, cell_type_path, cell_type_fraction_estimation_path, n_cells_per_spot_path, 
                st_cell_type_path, output_path, output_prefix, spaceranger_path=None, st_path=None, coordinates_path=None):
    if spaceranger_path is not None:
        st_data, coordinates_data = read_visium(spaceranger_path,output_path)
    elif (st_path is None) and (coordinates_path is None):
        raise ValueError("For ST data, you must provide either a tar.gz file or paths for expression and coordinates.")
    else:
        # Read data
        st_data = read_file(st_path)
        coordinates_data = read_file(coordinates_path)

    st_data = st_data[~st_data.index.duplicated(keep=False)]
    if (st_cell_type_path is None) and (cell_type_fraction_estimation_path is None):
        print('Estimating cell type fractions')
        if (spaceranger_path is not None):
            st_data_outpath = os.path.join(output_path, f"{output_prefix}ST_expression.txt")
            st_data.to_csv(st_data_outpath, sep='\t')
            cell_type_fraction_estimation_path = estimate_cell_type_fractions(scRNA_path, cell_type_path, st_data_outpath, output_path, output_prefix)
        else:
            cell_type_fraction_estimation_path = estimate_cell_type_fractions(scRNA_path, cell_type_path, st_path, output_path, output_prefix)

    st_data.columns = ['SPOT_'+str(col) for col in st_data.columns]
    st_data.index = ['GENE_'+str(idx) for idx in st_data.index]

    coordinates_data.index = ['SPOT_'+str(idx) for idx in coordinates_data.index]

    scRNA_data = read_file(scRNA_path)
    scRNA_data.columns = ['CELL_'+str(col) for col in scRNA_data.columns]
    scRNA_data.index = ['GENE_'+str(idx) for idx in scRNA_data.index]

    scRNA_data = scRNA_data[~scRNA_data.index.duplicated(keep=False)]

    cell_type_data = read_file(cell_type_path)
    cell_type_data.index = ['CELL_'+str(idx) for idx in cell_type_data.index]
    cell_type_data.iloc[:,0] = ['TYPE_'+str(cell) for cell in list(cell_type_data.iloc[:,0])]

    #st_data = st_data.loc[(st_data!=0).any(1), (st_data!=0).any(0)]
    #scRNA_data = scRNA_data.loc[(scRNA_data!=0).any(1), (scRNA_data!=0).any(0)]

    if st_cell_type_path is not None:
        st_cell_type_data = read_file(st_cell_type_path)
        st_cell_type_data.index = ['SPOT_'+str(idx) for idx in st_cell_type_data.index]
        st_cell_type_data.iloc[:,0] = ['TYPE_'+str(cell) for cell in list(st_cell_type_data.iloc[:,0])]
    else:
        st_cell_type_data = None
    if cell_type_fraction_estimation_path is not None:
        cell_type_fraction_data = read_file(cell_type_fraction_estimation_path)
        cell_type_fraction_data.columns = ['TYPE_'+str(col) for col in cell_type_fraction_data.columns]
    else:
        cell_type_fraction_data = None
    if n_cells_per_spot_path is not None:
        n_cells_per_spot_data = read_file(n_cells_per_spot_path)
        n_cells_per_spot_data.index = ['SPOT_'+str(idx) for idx in n_cells_per_spot_data.index]
    else:
        n_cells_per_spot_data = None

    # Order data to match
    try:
        st_data = st_data[coordinates_data.index]
        scRNA_data = scRNA_data[cell_type_data.index]
        if st_cell_type_data is not None:
            st_cell_type_data = st_cell_type_data.loc[coordinates_data.index, :]
        if n_cells_per_spot_data is not None:
            n_cells_per_spot_data = n_cells_per_spot_data.transpose(copy=False)
            n_cells_per_spot_data = n_cells_per_spot_data[coordinates_data.index]
            n_cells_per_spot_data = n_cells_per_spot_data.transpose(copy=False)
    except Exception as e:
        raise IndexError(f"The ST data: {st_path} and coordinates data: {coordinates_path} have to "
                         "have the same spot IDs for columns and rows, respectively, "
                         f"and scRNA data: {scRNA_path} and cell type data: {cell_type_path} have"
                         " to have the same cell IDs for columns and rows, respectively.")

    # Validate input
    if (st_data.columns != coordinates_data.index).any():
        raise IndexError(f"The ST data: {st_path} and coordinates data: {coordinates_path} have to "
                         "have the same spot IDs for columns and rows, respectively.")

    if (scRNA_data.columns != cell_type_data.index).any():
        raise IndexError(f"The scRNA data: {scRNA_path} and cell type data: {cell_type_path} have"
                         " to have the same cell IDs for columns and rows, respectively.")
    
    if (st_cell_type_data is not None) and (st_cell_type_data.index != coordinates_data.index).any():
        raise IndexError(f"The ST cell type data: {st_cell_type_path} and coordinates data: {coordinates_path} have to "
                         "have the same spot IDs for rows.")

    if (st_cell_type_data is None) and (cell_type_fraction_data is None):
        raise ValueError("At least one of st_cell_type_path and cell_type_fraction_estimation_path should be specified."
                         "For --single-cell, st_cell_type_path is recommended; if not --single-cell, cell_type_fraction_estimation_path is required.")
    
    if (st_cell_type_data is not None) and (cell_type_fraction_data is not None):
        print("Warning: st_cell_type_path and cell_type_fraction_estimation_path are both specified.")
        print("If --single-cell, cell_type_fraction_estimation_path will be ignored in this case.")

    return scRNA_data, cell_type_data, st_data, coordinates_data, cell_type_fraction_data, n_cells_per_spot_data, st_cell_type_data


def estimate_cell_number_RNA_reads(st_data, mean_cell_numbers):
    # Read data
    expressions = st_data.values.astype(float)

    # Data normalization
    expressions_tpm_log = normalize_data(expressions)

    # Set up fitting problem
    RNA_reads = np.sum(expressions_tpm_log, axis=0, dtype=float)
    mean_RNA_reads = np.mean(RNA_reads)
    min_RNA_reads = np.min(RNA_reads)

    min_cell_numbers = 1 if min_RNA_reads > 0 else 0

    fit_parameters = np.polyfit(np.array([min_RNA_reads, mean_RNA_reads]),
                                np.array([min_cell_numbers, mean_cell_numbers]), 1)
    polynomial = np.poly1d(fit_parameters)
    cell_number_to_node_assignment = polynomial(RNA_reads).astype(int)

    return cell_number_to_node_assignment


def get_cell_type_fraction(number_of_cells, cell_type_fraction_data):
    # Uncomment commented lines for closer numbers
    cell_type_numbers = cell_type_fraction_data.transpose()
    #fractions_to_numbers = np.round(cell_type_numbers.values*number_of_cells)
    fractions_to_numbers = cell_type_numbers.values*number_of_cells
    fractions_to_numbers = fractions_to_numbers.round().astype(int)
    cell_type_numbers.iloc[:,0] = fractions_to_numbers
    max_ct = cell_type_numbers.idxmax().values[0]
    return cell_type_numbers


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
    print(cell_type_numbers_int.values[:,0])
    for cell_type in unique_cell_type_labels:
        cell_type_index = np.nonzero(cell_type_data.values[:, 0] == cell_type)[0].tolist()
        cell_type_count_available = len(cell_type_index)
        if cell_type_count_available == 0:
            raise ValueError(f"Cell type {cell_type} in the ST dataset is not available in the scRNA-seq dataset.")
        cell_type_count_desired = cell_type_numbers_int.loc[cell_type][0]


        if sampling_method == "place_holders":
            if cell_type_count_desired > cell_type_count_available:
                num_genes = scRNA_data.shape[0]
                num_placeholder_cells = cell_type_count_desired - cell_type_count_available
                scRNA_original_np = scRNA_data.iloc[:, cell_type_index].to_numpy()
                scRNA_placeholder_np = np.zeros((num_genes, num_placeholder_cells))

                sampled_index = np.random.choice(cell_type_index, num_placeholder_cells)

                for i1 in range(num_placeholder_cells):
                    scRNA_placeholder_np[:, i1] = [np.random.choice(scRNA_original_np[j1, :]) for j1 in range(num_genes)]

                # # alternate implementation (vectorization on one axis)
                # for gene_idx in range(num_genes):
                #     rand_idxs = np.random.randint(0, cell_type_count_available, size=num_placeholder_cells)
                #     scRNA_placeholder_np[gene_idx, :] = scRNA_original_np[gene_idx, rand_idxs.tolist()]
                
                cell_names_list.append(np.array(scRNA_data.columns.values[cell_type_index]))
                cell_names_list.append(np.array([cell_type.replace('TYPE_', 'CELL_') + '_new_' + str(i+1) for i in range(num_placeholder_cells)]))
                all_cells_save_list.append(scRNA_original_np)
                all_cells_save_list.append(scRNA_placeholder_np)

            else:
                cell_type_selected_index = random.sample(cell_type_index, cell_type_count_desired)

                cell_names_list.append(scRNA_data.columns.values[cell_type_selected_index])
                all_cells_save_list.append(scRNA_data.iloc[:, cell_type_selected_index].to_numpy())

        elif sampling_method == "duplicates":
            if cell_type_count_desired > cell_type_count_available:
                cell_type_selected_index = np.concatenate([
                    cell_type_index, np.random.choice(cell_type_index, cell_type_count_desired - cell_type_count_available)
                ], axis=0) # ensure at least one copy of each, then sample the rest

            else:
                cell_type_selected_index = random.sample(cell_type_index, cell_type_count_desired)
        
            sampled_index_total.append(cell_type_selected_index)
        
        else:
            raise ValueError("Invalid sampling_method provided")
    
    if sampling_method == "place_holders":
        all_cells_save = pd.DataFrame(
                            np.concatenate(all_cells_save_list, axis=1),
                            index=scRNA_data.index,
                            columns=np.concatenate(cell_names_list, axis=0)
        )
    else:
        sampled_index_total = np.concatenate(sampled_index_total, axis=0).astype(int)
        all_cells_save = scRNA_data.iloc[:, sampled_index_total]

    return all_cells_save


def solve_linear_assignment_problem(scRNA_norm_data, st_norm_data, cell_number_to_node_assignment,
                                    solver_method, solver, seed, distance_metric, process_idx=None):
    """
    Parameters :
        scRNA_norm_data (2D np.ndarray(float)) : normalized gene x cell scRNA expression data to be used as reference
        st_norm_data    (2D np.ndarray(float)) : normalized gene x cell ST expression data to be used as target
        cell_number_to_node_assignment (1D np.ndarray(int)) :
            estimated cell count at each ST spot, where the value at index n denotes the cell count at the nth spot of st_norm_data
        solver_method, solver, seed, distance_metric : as specified in the input args
        process_idx           (int) : returned as is; used to keep track of asynchronous processes in apply_linear_assignment
    Returns :
        mapped_st_index (List[int]) : indices of ST spots (column index of st_norm_data) where each single cell is mapped.
                                        list has length scRNA_norm_data.shape[1]; order of single cells follows scRNA_norm_data.
        process_idx           (int) : returned as is.
    """
    print(scRNA_norm_data.shape)
    print(st_norm_data.shape)
    distance_repeat, location_repeat =\
        calculate_cost(scRNA_norm_data, st_norm_data, cell_number_to_node_assignment,
                       solver_method, distance_metric)
    print(distance_repeat)
    if solver_method == 'lapjv' or solver_method == 'lapjv_compat':
        print('Solving linear assignment problem ...')
        np.random.seed(seed)
        cost_scaled = distance_repeat + 1e-16 * np.random.rand(distance_repeat.shape[0],
                                                               distance_repeat.shape[1])
        t0 = time.perf_counter()
        assignment = call_solver(solver, solver_method, cost_scaled)
        print(f"Time to solve linear assignment problem: {round(time.perf_counter() - t0, 2)} seconds")
        assigned_nodes = location_repeat[assignment]
        mapped_st_index = np.transpose(assigned_nodes).tolist()

    elif solver_method == 'lap_CSPR':
        print('Solving linear assignment problem ...')
        np.random.seed(seed)
        cost_scaled = 10**6 * distance_repeat + 10 * np.random.rand(distance_repeat.shape[0],
                                                                    distance_repeat.shape[1]) + 1
        cost_scaled = np.transpose(cost_scaled)
        cost_scaled_int = cost_scaled.astype(int)
        cost_scaled_int_list = cost_scaled_int.tolist()
        t0 = time.perf_counter()
        assignment = match_solution(cost_scaled_int_list)

        print(f"Time to solve linear assignment problem: {round(time.perf_counter() - t0, 2)} seconds")
        assigned_nodes = location_repeat[assignment[:, 0].astype(int)]
        mapped_st_index = assigned_nodes.tolist()
    else:
        raise ValueError("Invalid solver_method provided")

    return mapped_st_index, process_idx


def apply_linear_assignment(scRNA_data, st_data, coordinates_data, cell_number_to_node_assignment,
                                solver_method, solver, seed, distance_metric, number_of_processors,
                                index_sc_list, index_st_list=None, subsampled_cell_number_to_node_assignment_list=None):
    """
    Parallelizes process by queueing a subprocess for each subset.
    Each subset is specified by index_st_list or subsampled_cell_number_to_node_assignment_list (in reference to st_data)
        along with index_sc_list (in reference to sc_data).
    The output of each subprocess is aggregated and returned as a single object.

    Parameters :
        scRNA_data, st_data, coordinates_data (pd.DataFrame) :
            formatted as read in from read_data().
            scRNA_data will have been sampled upstream.
        cell_number_to_node_assignment (1D np.ndarray(int)) :
            estimated cell count at each ST spot, where the value at index n denotes the cell count at the nth spot of st_data
        solver_method, solver, seed, distance_metric, number_of_processors :
            as specified in the input args.

        index_sc_list (List[1D np.array(int)]) : 
            partition of scRNA_data cell indices (each of length number_of_selected_(sub_)spots, with the exception of the last),
            that denotes the subsets that scRNA_data will be partitioned into for parallelization.

        index_st_list (List[1D np.array(int)]) :
            used if --single-cell.
            partition of st_data spot indices (each of length number_of_selected_spots, with the exception of the last partition),
            that denotes the subsets that st_data will be partitioned into for parallelization.
        subsampled_cell_number_to_node_assignment_list (List[1D np.ndarray(int)]) :
            used if --sampling-sub-spots.
            list of cell_number_to_node_assignment (each of length [spot count of ST data] and
                summing to number_of_selected_sub_spots with the exception of the last),
            where each is used for a subprocess mapping (sampled) scRNA_data to st_data.

    Returns :
        assigned_locations (pd.DataFrame) :
            list of ST coordinates (from coordinates_data) where each single cell in cell_ids_selected is mapped to.
            index = Spot ID; columns = coordinates_data.columns
        cell_ids_selected (1D np.array) :
            list of single cell IDs (from scRNA_data.index; with duplicates if necessary) that are mapped.
            the nth single cell in cell_ids_selected is mapped to the spot on the nth row of assigned_locations.
    """
    if (index_st_list is not None) and (subsampled_cell_number_to_node_assignment_list is not None):
        raise ValueError("index_st_list and subsampled_cell_number_to_node_assignment_list cannot both be specified")
    
    # normalize data; output is an np.ndarray
    scRNA_norm_np = normalize_data(scRNA_data.to_numpy())
    st_norm_np = normalize_data(st_data.to_numpy())

    # regenerate pandas dataframe from the normalized data
    scRNA_norm_data = pd.DataFrame(scRNA_norm_np, index=scRNA_data.index, columns=scRNA_data.columns)
    st_norm_data = pd.DataFrame(st_norm_np, index=st_data.index, columns=st_data.columns)

    if (index_st_list is None) and (subsampled_cell_number_to_node_assignment_list is None):
        mapped_st_index, _ =\
            solve_linear_assignment_problem(
                scRNA_norm_data.iloc[:, index_sc_list[0]].to_numpy(), st_norm_data.to_numpy(), cell_number_to_node_assignment,
                    solver_method, solver, seed, distance_metric)
        assigned_locations = coordinates_data.iloc[mapped_st_index]
        cell_ids_selected = scRNA_norm_data.columns.values[index_sc_list[0]]

        return assigned_locations, cell_ids_selected

    results = []
    cell_ids_selected_list = [] # List of 1D np.array (single cell ID) from each process
    assigned_locations_list = [] # List of pd.DataFrame (ST spot coordinates) from each process

    # compute the number of processes
    if index_st_list is not None:
        # called for --single_cell
        num_iters = len(index_st_list)
    elif subsampled_cell_number_to_node_assignment_list is not None:
        # called for --sampling-sub-spots
        num_iters = len(subsampled_cell_number_to_node_assignment_list)
    else:
        raise ValueError("Invalid point")
    print(f"Number of required processors: {num_iters}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=min(num_iters, number_of_processors)) as executor:
        for idx in range(num_iters):
            if index_st_list is not None:
                # called for --single-cell
                st_norm_data_selected = st_norm_data.iloc[:, index_st_list[idx]]
                cell_number_to_node_assignment_selected = cell_number_to_node_assignment[index_st_list[idx]]
            elif subsampled_cell_number_to_node_assignment_list is not None:
                # called for --sampling-sub-spots
                st_norm_data_selected = st_norm_data
                cell_number_to_node_assignment_selected = subsampled_cell_number_to_node_assignment_list[idx]
            else:
                raise ValueError("Invalid point")

            scRNA_norm_data_selected = scRNA_norm_data.iloc[:, index_sc_list[idx]]
            
            # launch process
            results.append(executor.submit(
                solve_linear_assignment_problem,
                    scRNA_norm_data_selected.to_numpy(), st_norm_data_selected.to_numpy(), cell_number_to_node_assignment_selected,
                    solver_method, solver, seed, distance_metric, process_idx=idx
                )
            )
        
        for f in concurrent.futures.as_completed(results):
            # aggregate results
            mapped_st_index, process_idx = f.result()

            # because the nth single cell in cell_ids_selected is mapped to the nth spot in assigned_locations,
            # the orders of assigned_locations and cell_ids_selected should match.
            assigned_locations = coordinates_data.iloc[index_st_list[process_idx]].iloc[mapped_st_index]\
                                    if index_st_list is not None \
                                    else coordinates_data.iloc[mapped_st_index]
            assigned_locations_list.append(assigned_locations)
            cell_ids_selected = scRNA_norm_data.columns.values[index_sc_list[process_idx]]
            cell_ids_selected_list.append(cell_ids_selected)
        
        cell_ids_selected = np.concatenate(cell_ids_selected_list, axis=0)
        assigned_locations = pd.concat(assigned_locations_list)

    return assigned_locations, cell_ids_selected