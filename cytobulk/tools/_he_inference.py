import torch
from torchvision import datasets, transforms
from torch.utils import data
import imageio
import sys
from os.path import exists
import numpy as np
import pandas as pd
import scanpy as sc
import os
from skimage.measure import label, regionprops
from .model import DeepCMorph
from .. import get
from .. import utils
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2

#np.random.seed(42)

# Modify the target number of classes and the path to the dataset
#NUM_CLASSES = 41
#PATH_TO_SAMPLE_FOLDER = "data/sample_TCGA_images/"

# Fix binom_test import issue
try:
    from scipy.stats import binomtest
    def binom_test(x, n, p, alternative='two-sided'):
        return binomtest(x, n, p, alternative=alternative).pvalue
except ImportError:
    try:
        from scipy.stats import binom_test
    except ImportError:
        from scipy.stats import binom
        def binom_test(x, n, p, alternative='two-sided'):
            if alternative == 'greater':
                return 1 - binom.cdf(x - 1, n, p)
            elif alternative == 'less':
                return binom.cdf(x, n, p)
            else:
                return 2 * min(binom.cdf(x, n, p), 1 - binom.cdf(x - 1, n, p))
            

def process_txt_files(input_dir, output_file_name, project_name_filter=None):
    output_file_name=output_file_name+".txt"

    output_file_path = os.path.join(input_dir, output_file_name)

    # 打开输出文件
    with open(output_file_path, "w") as outfile:

        for filename in os.listdir(input_dir):
            if filename.endswith(".txt"):

                parts = filename.split("_")
                if len(parts) < 4 or not parts[-1].endswith("cent.txt"):
                    continue
                
                project_name = "_".join(parts[:-4])  
                x_offset = int(parts[-4])
                y_offset = int(parts[-3])
                

                if project_name_filter and project_name != project_name_filter:
                    continue

                input_path = os.path.join(input_dir, filename)
                

                with open(input_path, "r") as infile:
                    for line in infile:
                        line = line.strip()
                        if not line:  
                            continue

                        try:
                            x, y, label = line.split("\t")
                            x = int(x) + x_offset
                            y = int(y) + y_offset
                            
                            outfile.write(f"{project_name}\t{x}\t{y}\t{label}\n")
                        except Exception:
                            pass

    print(f"Data successfully written to {output_file_path}")

def inference_cell_type_from_he_image(
        image_dir,
        out_dir,
        project):
    if exists(f'{out_dir}/combinded_cent.txt'):
        df = pd.read_csv(f'{out_dir}/combinded_cent.txt',sep='\t',header=None)
        df.columns = ['data_set','x','y','cell_type']
        print("print(f'{out_dir}/combinded_cent.txt already exists, skipping prediction.')")
    else:
        NUM_CLASSES = 41
        PATH_TO_SAMPLE_FOLDER = image_dir + '/'
        utils.check_paths(f'{out_dir}')

        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = torch.device("cuda")

        # Defining the model
        model = DeepCMorph(num_classes=NUM_CLASSES)
        # Loading model weights corresponding to the TCGA Pan Cancer dataset
        # Possible dataset values:  TCGA, TCGA_REGULARIZED, CRC, COMBINED
        model.load_weights(dataset="COMBINED")

        model.to(device)
        model.eval()

        # Loading test images
        test_transforms = transforms.Compose([transforms.ToTensor()])
        test_dataset = datasets.ImageFolder(PATH_TO_SAMPLE_FOLDER, transform=test_transforms)
        class_names = test_dataset.classes
        test_dataloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=False, drop_last=False)
        TEST_SIZE = len(test_dataloader.dataset)
        
        print("Generating segmentation and classification maps for sample images")

        with torch.no_grad():

            feature_maps = np.zeros((TEST_SIZE, 2560))

            image_id = 0

            test_iter = iter(test_dataloader)
            for j in range(len(test_dataloader)):
                image, labels = next(test_iter)
                image = image.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                names = class_names[j]

                # Get predicted segmentation and classification maps for each input images
                nuclei_segmentation_map, nuclei_classification_maps = model(image, return_segmentation_maps=True)
                
                nuclei_segmentation_map_for_central = nuclei_segmentation_map.detach().cpu().numpy()[0]
                # find nuclei central
                binary_image = (nuclei_segmentation_map_for_central > 0.5).astype(np.uint8)
                labeled_image = label(binary_image)
                regions = regionprops(labeled_image)
                
                label_mapping = {
                    1: "Lymphocytes",
                    2: "Epithelial Cells",
                    3: "Plasma Cells",
                    4: "Neutrophils",
                    5: "Eosinophils",
                    6: "Connective Tissue"
                }

                centroids = []
                
                
                for region in regions:
                    centroid = region.centroid
                    y, x = int(centroid[1]), int(centroid[2])  
                    label_value = nuclei_classification_maps.detach().cpu().numpy()[0].argmax(axis=0)[y, x]
                    cell_type = label_mapping.get(label_value)
                    if cell_type is not None:  
                        centroids.append((x, y, cell_type))  # (x, y, cell_type)

                # Visualizing the predicted segmentation map
                nuclei_segmentation_map = nuclei_segmentation_map.detach().cpu().numpy()[0].transpose(1,2,0) * 255
                nuclei_segmentation_map = np.dstack((nuclei_segmentation_map, nuclei_segmentation_map, nuclei_segmentation_map))

                # Visualizing the predicted nuclei classification map
                nuclei_classification_maps = nuclei_classification_maps.detach().cpu().numpy()[0].transpose(1, 2, 0)
                nuclei_classification_maps = np.argmax(nuclei_classification_maps, axis=2)

                nuclei_classification_maps_visualized = np.zeros((nuclei_classification_maps.shape[0], nuclei_classification_maps.shape[1], 3))
                nuclei_classification_maps_visualized[nuclei_classification_maps == 1] = [255, 0, 0]
                nuclei_classification_maps_visualized[nuclei_classification_maps == 2] = [0, 255, 0]
                nuclei_classification_maps_visualized[nuclei_classification_maps == 3] = [0, 0, 255]
                nuclei_classification_maps_visualized[nuclei_classification_maps == 4] = [255, 255, 0]
                nuclei_classification_maps_visualized[nuclei_classification_maps == 5] = [255, 0, 255]
                nuclei_classification_maps_visualized[nuclei_classification_maps == 6] = [0, 255, 255]
                
                import matplotlib.pyplot as plt

                image = image.detach().cpu().numpy()[0].transpose(1,2,0)
                plt.imshow(image, cmap='gray')
                plt.imshow(nuclei_classification_maps_visualized.astype(np.uint8), alpha=0.5)  # ?????
                for cent in centroids:
                    plt.plot(cent[0], cent[1], 'ko',markersize=5)
                plt.title("centroids")
                plt.axis('off') 
                

                plt.savefig(f"{out_dir}/{project}_{names}" + 'cell_centroids.png', format='png', bbox_inches='tight', dpi=300)
                plt.close() 
                
                cent_data = {
                    "x": [cent[0] for cent in centroids],
                    "y": [cent[1] for cent in centroids],
                    "label": [cent[2] for cent in centroids]
                }
                df = pd.DataFrame(cent_data)
                df.to_csv(f"{out_dir}/{project}_{names}_cell_cent.txt", sep="\t", index=False)

                image=image * 255
                # Saving visual results
                combined_image = np.hstack((image, nuclei_segmentation_map, nuclei_classification_maps_visualized))
                imageio.imsave(f"{out_dir}/{project}_{names}" + ".jpg", combined_image.astype(np.uint8))
                image_id += 1

            print("All visual results saved")

            print("Combine results")
            process_txt_files(out_dir, "combinded_cent", project_name_filter=None)
            df = pd.read_csv(f'{out_dir}/combinded_cent.txt',sep='\t',header=None)
            df.columns = ['data_set','x','y','cell_type']
            print("Save file done")
    return df




def load_graph1(cell_data, k=15):
    from scipy.spatial.distance import cdist
    import numpy as np
    
    coordinates = cell_data[['x', 'y']].values
    labels = cell_data['cell_type'].values
    distances = cdist(coordinates, coordinates, metric="euclidean")
    
    adjacency_matrix = np.ones_like(distances)
    
    for i in range(len(distances)):

        nearest_indices = np.argsort(distances[i])[:k + 1]
        nearest_distances = distances[i][nearest_indices]
        

        adjacency_matrix[i, i] = 0
        

        neighbor_indices = nearest_indices[1:] 
        neighbor_distances = nearest_distances[1:]  
        
        if len(neighbor_distances) > 0:

            max_distance = neighbor_distances[-1]  
            
            if max_distance > 0:

                normalized_values = neighbor_distances / max_distance
            else:

                normalized_values = np.zeros_like(neighbor_distances)
            

            adjacency_matrix[i, neighbor_indices] = normalized_values
    
    return adjacency_matrix, labels

def compute_gene_expression_similarity(he_coords, anchor_adata, sc_adata, shared_genes, chunk_size=2000):

    anchor_expr = get.count_data_t(anchor_adata[:, shared_genes])
    sc_expr = get.count_data_t(sc_adata[:, shared_genes])
    

    he_coords_array = he_coords[['x', 'y']].values
    anchor_coords_array = anchor_adata.obsm['spatial']
    
    print("Finding nearest anchor points...")
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(he_coords_array, anchor_coords_array)
    nearest_anchor_indices = np.argmin(dist_matrix, axis=1)
    

    he_expr_matrix = anchor_expr.iloc[nearest_anchor_indices].values
    sc_expr_matrix = sc_expr.values

    n_he_cells = len(he_coords)
    n_sc_cells = len(sc_adata)


    similarity_matrix = np.zeros((n_he_cells, n_sc_cells))
    
    print(f"Computing MSE in chunks...")

    for i in range(0, n_he_cells, chunk_size):
        end_idx = min(i + chunk_size, n_he_cells)
        chunk_he_expr = he_expr_matrix[i:end_idx]  # (chunk_size, n_genes)
        

        chunk_he_expanded = chunk_he_expr[:, np.newaxis, :]  # (chunk_size, 1, n_genes)
        sc_expr_expanded = sc_expr_matrix[np.newaxis, :, :]   # (1, n_sc_cells, n_genes)
        
        chunk_mse = np.mean((chunk_he_expanded - sc_expr_expanded) ** 2, axis=2)
        similarity_matrix[i:end_idx, :] = chunk_mse
        
        if (i // chunk_size + 1) % 10 == 0:
            print(f"Processed {i + chunk_size}/{n_he_cells} HE cells...")
    
    print(f"MSE matrix computed with shape: {similarity_matrix.shape}")
    return similarity_matrix

def construct_cost_matrix_with_expression(labels1, labels2, expression_similarity=None, 
                                        mismatch_penalty=1000, expression_weight=1):

    print(f"Computing cost matrix for {len(labels1)} x {len(labels2)} cell pairs...")
    

    labels1_array = np.array(labels1)
    labels2_array = np.array(labels2)
    

    labels1_expanded = labels1_array[:, np.newaxis]  # (n_he_cells, 1)
    labels2_expanded = labels2_array[np.newaxis, :]  # (1, n_sc_cells)
    

    type_match_matrix = (labels1_expanded == labels2_expanded)  # (n_he_cells, n_sc_cells)
    

    type_cost_matrix = np.where(type_match_matrix, 0, mismatch_penalty)

    if expression_similarity is None:
        print("Using only cell type matching costs")
        return type_cost_matrix
    

    print("Combining cell type and expression similarity costs...")

    max_mse = np.max(expression_similarity)
    min_mse = np.min(expression_similarity)
    
    if max_mse > min_mse:
        normalized_mse = (expression_similarity - min_mse) / (max_mse - min_mse)
    else:
        normalized_mse = np.zeros_like(expression_similarity)
    

    cost_matrix = (1 - expression_weight) * type_cost_matrix + expression_weight * normalized_mse
    
    print(f"Cost matrix computed with shape: {cost_matrix.shape}")
    return cost_matrix



def compute_cosine_similarity_matrix(ligand_df, receptor_df):
    """
    Compute cell-cell cosine similarity between two gene expression matrices
    using matrix operations.

    Parameters:
    - ligand_df: DataFrame, rows are cells, columns are ligand genes.
    - receptor_df: DataFrame, rows are cells, columns are receptor genes.

    Returns:
    - similarity_matrix: DataFrame, cell-cell cosine similarity matrix.
    """
    # Ensure the indices of ligand_df and receptor_df match
    if not ligand_df.index.equals(receptor_df.index):
        raise ValueError("The indices of ligand_df and receptor_df (cells) must match.")
    
    # Convert DataFrames to numpy arrays
    ligand_matrix = ligand_df.values  # shape: (cells, ligand_genes)
    receptor_matrix = receptor_df.values  # shape: (cells, receptor_genes)

    # Compute dot product between all pairs of cells
    dot_product = np.dot(ligand_matrix, receptor_matrix.T)  # shape: (cells, cells)

    # Compute norms (L2 norm) for each cell in ligand and receptor matrices
    ligand_norms = np.linalg.norm(ligand_matrix, axis=1, keepdims=True)  # shape: (cells, 1)
    receptor_norms = np.linalg.norm(receptor_matrix, axis=1, keepdims=True)  # shape: (cells, 1)

    # Compute cosine similarity: dot_product / (ligand_norms * receptor_norms.T)
    similarity_matrix = dot_product / (ligand_norms * receptor_norms.T)

    # Convert to DataFrame for better readability
    similarity_matrix = pd.DataFrame(similarity_matrix, index=ligand_df.index, columns=ligand_df.index)
    return similarity_matrix

def compute_pearson_correlation(cell_matrix):
    """
    Compute cell-cell Pearson correlation using matrix operations.

    Parameters:
    - matrix: DataFrame, rows are cells, columns are genes (cell × gene).

    Returns:
    - correlation_matrix: DataFrame, cell-cell Pearson correlation matrix.
    """
    # Subtract mean (center the data)
    matrix = cell_matrix.values
    centered_matrix = matrix - matrix.mean(axis=1, keepdims=True)

    # Normalize rows (divide by standard deviation)
    norm = np.linalg.norm(centered_matrix, axis=1, keepdims=True)
    normalized_matrix = centered_matrix / norm

    # Compute Pearson correlation as the dot product of normalized rows
    correlation_matrix = np.dot(normalized_matrix, normalized_matrix.T)

    # Convert to DataFrame for better readability
    correlation_matrix = pd.DataFrame(correlation_matrix, index=cell_matrix.index, columns=cell_matrix.index)
    return correlation_matrix

def compute_weighted_distance_matrix(
    lr_affinity_matrix_part1,
    lr_affinity_matrix_part2,
    gene_correlation_matrix,
    cell_type,
    alpha_same_type = 0.2,
    alpha_diff_type = 1
):
    """
    Compute a weighted distance matrix based on the given affinity matrices and cell types.

    Parameters:
    - lr_affinity_matrix_part1: DataFrame, cell-cell affinity matrix (part 1).
    - lr_affinity_matrix_part2: DataFrame, cell-cell affinity matrix (part 2).
    - gene_correlation_matrix: DataFrame, cell-cell correlation matrix.
    - cell_type: Series or array, cell types corresponding to the index of the matrices.

    Returns:
    - distance_matrix: DataFrame, the weighted distance matrix.
    """
    # Ensure all matrices are aligned
    if not (lr_affinity_matrix_part1.index.equals(lr_affinity_matrix_part2.index) and
            lr_affinity_matrix_part1.index.equals(gene_correlation_matrix.index)):
        raise ValueError("All matrices must have the same index and column order.")
    
    # Convert cell_type to a Series if it's not already
    if not isinstance(cell_type, pd.Series):
        cell_type = pd.Series(cell_type, index=lr_affinity_matrix_part1.index)
    
    # Get the number of cells
    num_cells = len(cell_type)

    # Compute the combined ligand-receptor affinity matrix
    print("without_abs")
    lr_affinity_sum = (lr_affinity_matrix_part1 + lr_affinity_matrix_part2)/2


    # Initialize alpha matrix
    alpha_matrix = np.zeros((num_cells, num_cells))
    cell_type_values = cell_type.values


    # Fill in alpha matrix
    for i in range(num_cells):
        for j in range(num_cells):
            if cell_type_values[i] == cell_type_values[j]:
                alpha_matrix[i, j] = alpha_same_type  # Same type
            else:
                alpha_matrix[i, j] = alpha_diff_type  # Different type

    # Convert alpha matrix to DataFrame for alignment with other matrices
    alpha_matrix = pd.DataFrame(alpha_matrix, index=lr_affinity_sum.index, columns=lr_affinity_sum.columns)

    # Compute the weighted distance matrix
    weighted_matrix = (
        alpha_matrix * lr_affinity_sum + 
        (1 - alpha_matrix) * gene_correlation_matrix
    )

    distance_matrix = 1 - weighted_matrix

    return distance_matrix


def load_graph2_with_LR_affinity(adata, graph1_labels,lr_data,annotation_key="celltype_minor"):
    print("sample single cells according to predicted label")
    sampled_cells = []
    for cell_type in np.unique(graph1_labels):
        matching_cells = adata[adata.obs[annotation_key] == cell_type].obs_names
        num_cells = np.sum(graph1_labels == cell_type)
        if len(matching_cells) < num_cells:
            sampled_cells.extend(matching_cells[np.random.choice(len(matching_cells), num_cells, replace=True)])
        else:
            sampled_cells.extend(matching_cells[np.random.choice(len(matching_cells), num_cells, replace=False)])
    sampled_adata = adata[sampled_cells,:]
    sampled_adata.obs_names_make_unique()
    print("compute LR affinity")
    expression_data = get.count_data_t(sampled_adata)
    ligand_matrix = expression_data.loc[:, lr_data['ligand'].values]  
    receptor_matrix = expression_data.loc[:, lr_data['receptor'].values]
    lr_affinity_matrix_part1 = compute_cosine_similarity_matrix(ligand_matrix, receptor_matrix)
    lr_affinity_matrix_part2 = compute_cosine_similarity_matrix(receptor_matrix, ligand_matrix)
    gene_correlation_matrix = compute_pearson_correlation(expression_data)
    distance_matrix = compute_weighted_distance_matrix(
        lr_affinity_matrix_part1, 
        lr_affinity_matrix_part2, 
        gene_correlation_matrix, 
        sampled_adata.obs[annotation_key].values)
    return distance_matrix, sampled_adata.obs[annotation_key].values,sampled_adata,sampled_cells


def construct_cost_matrix(labels1, labels2, mismatch_penalty=1000):
    cost_matrix = np.zeros((len(labels1), len(labels2)))
    for i, label1 in enumerate(labels1):
        for j, label2 in enumerate(labels2):
            cost_matrix[i, j] = 0 if label1 == label2 else mismatch_penalty
    return cost_matrix

def extract_matching_relationships(gw_trans, locations, cells):
    matches = []
    for i in range(gw_trans.shape[0]): 

        matched_cell_index = np.argmax(gw_trans[i, :])
        matched_cell = cells[matched_cell_index]
        matches.append((locations[i], matched_cell))
    
    return matches




def spatial_partition(cell_coordinates, max_cells_per_batch=20000):
    """
    Partition cell coordinates into multiple continuous regions based on spatial location,
    ensuring each region does not exceed the specified number of cells
    
    Parameters:
    - cell_coordinates: DataFrame with columns ['x', 'y', 'cell_type']
    - max_cells_per_batch: Maximum number of cells per batch
    
    Returns:
    - List of DataFrames, each containing cells for one batch
    """
    if len(cell_coordinates) <= max_cells_per_batch:
        return [cell_coordinates]
    
    # Calculate required number of batches
    n_batches = int(np.ceil(len(cell_coordinates) / max_cells_per_batch))
    
    # Get coordinate ranges
    x_min, x_max = cell_coordinates['x'].min(), cell_coordinates['x'].max()
    y_min, y_max = cell_coordinates['y'].min(), cell_coordinates['y'].max()
    
    # Calculate grid partitioning
    # Try to create approximately square grids
    aspect_ratio = (x_max - x_min) / (y_max - y_min) if (y_max - y_min) > 0 else 1
    n_cols = int(np.ceil(np.sqrt(n_batches * aspect_ratio)))
    n_rows = int(np.ceil(n_batches / n_cols))
    
    # Calculate grid size
    x_step = (x_max - x_min) / n_cols if n_cols > 0 else x_max - x_min
    y_step = (y_max - y_min) / n_rows if n_rows > 0 else y_max - y_min
    
    batches = []
    
    for i in range(n_rows):
        for j in range(n_cols):
            if len(batches) >= n_batches:
                break
                
            # Define current grid boundaries
            x_start = x_min + j * x_step
            x_end = x_min + (j + 1) * x_step if j < n_cols - 1 else x_max + 1  # +1 to include boundary
            y_start = y_min + i * y_step
            y_end = y_min + (i + 1) * y_step if i < n_rows - 1 else y_max + 1  # +1 to include boundary
            
            # Select cells within current grid
            mask = (
                (cell_coordinates['x'] >= x_start) & 
                (cell_coordinates['x'] < x_end) &
                (cell_coordinates['y'] >= y_start) & 
                (cell_coordinates['y'] < y_end)
            )
            
            batch_cells = cell_coordinates[mask].copy()
            
            if len(batch_cells) > 0:
                # If this batch is too large, further subdivide
                if len(batch_cells) > max_cells_per_batch:
                    sub_batches = spatial_partition(batch_cells, max_cells_per_batch)
                    batches.extend(sub_batches)
                else:
                    batches.append(batch_cells)
    
    # Handle remaining cells (if any)
    assigned_indices = set()
    for batch in batches:
        assigned_indices.update(batch.index)
    
    remaining_cells = cell_coordinates[~cell_coordinates.index.isin(assigned_indices)]
    if len(remaining_cells) > 0:
        if len(remaining_cells) <= max_cells_per_batch:
            batches.append(remaining_cells)
        else:
            # If too many remaining cells, distribute them to existing batches
            remaining_cells_list = remaining_cells.index.tolist()
            for i, cell_idx in enumerate(remaining_cells_list):
                if len(batches) > 0:
                    batch_idx = i % len(batches)
                    batches[batch_idx] = pd.concat([batches[batch_idx], remaining_cells.loc[[cell_idx]]])
                else:
                    batches.append(remaining_cells.loc[[cell_idx]])
    
    return batches

def downsample_sc_data(sc_adata, target_cell_types, annotation_key="curated_celltype"):
    """
    Downsample single-cell data based on target cell types
    
    Parameters:
    - sc_adata: AnnData object
    - target_cell_types: dict, {cell_type: count} target cell types and their counts
    - annotation_key: str, cell type annotation key
    
    Returns:
    - Downsampled AnnData object
    """
    sampled_cells = []
    
    for cell_type, target_count in target_cell_types.items():
        # Get all cells of this cell type
        type_cells = sc_adata[sc_adata.obs[annotation_key] == cell_type].obs_names
        
        if len(type_cells) == 0:
            print(f"Warning: No cells found for cell type {cell_type}")
            continue
            
        # If target count exceeds available count, use sampling with replacement
        if target_count > len(type_cells):
            sampled_indices = np.random.choice(len(type_cells), target_count, replace=True)
            sampled_cells.extend(type_cells[sampled_indices])
        else:
            # Otherwise use sampling without replacement
            sampled_indices = np.random.choice(len(type_cells), target_count, replace=False)
            sampled_cells.extend(type_cells[sampled_indices])
    
    # Create downsampled data
    downsampled_adata = sc_adata[sampled_cells, :].copy()
    downsampled_adata.obs_names_make_unique()
    
    return downsampled_adata,sampled_cells



def identify_spatially_biased_cell_types(coord_df, grid_size=(6, 6), 
                                       presence_threshold=0.05, 
                                       ubiquity_threshold=0.75,
                                       min_cells_per_grid=5):
    """
    Step 1: Identify and exclude ubiquitous cell types, keep only spatially biased ones
    """
    print("=== Step 1: Identify Spatially Biased Cell Types ===\n")
    
    # Create spatial grid
    x_min, x_max = coord_df['x'].min(), coord_df['x'].max()
    y_min, y_max = coord_df['y'].min(), coord_df['y'].max()
    
    x_bins = np.linspace(x_min, x_max, grid_size[0] + 1)
    y_bins = np.linspace(y_min, y_max, grid_size[1] + 1)
    
    coord_df_temp = coord_df.copy()
    coord_df_temp['grid_x'] = pd.cut(coord_df_temp['x'], x_bins, labels=False, include_lowest=True)
    coord_df_temp['grid_y'] = pd.cut(coord_df_temp['y'], y_bins, labels=False, include_lowest=True)
    coord_df_temp['grid_id'] = coord_df_temp['grid_x'] * grid_size[1] + coord_df_temp['grid_y']
    
    ubiquitous_types = []
    focal_types = []
    
    for cell_type in coord_df['cell_type'].unique():
        
        valid_grids = 0
        present_grids = 0
        
        for grid_id in coord_df_temp['grid_id'].unique():
            grid_cells = coord_df_temp[coord_df_temp['grid_id'] == grid_id]
            
            if len(grid_cells) >= min_cells_per_grid:  # Valid grid
                valid_grids += 1
                type_count = len(grid_cells[grid_cells['cell_type'] == cell_type])
                type_proportion = type_count / len(grid_cells)
                
                if type_proportion >= presence_threshold:
                    present_grids += 1
        
        # Calculate ubiquity score
        ubiquity_score = present_grids / valid_grids if valid_grids > 0 else 0
        
        # Classify
        if ubiquity_score >= ubiquity_threshold:
            category = "Ubiquitous (Exclude)"
            ubiquitous_types.append(cell_type)
        else:
            category = "Spatially Biased (Analyze)"
            focal_types.append(cell_type)
        

    
    return ubiquitous_types, focal_types

def calculate_clustering_score(coord_df, focal_types,
                             spatial_significance_threshold=0.05,
                             autocorrelation_threshold=0.1,
                             grid_size=(5, 5)):
    """
    Step 2: Calculate clustering score for spatially biased cell types only
    Score range: 0.0 - 0.5
    """
    
    if len(focal_types) == 0:

        return {
            'clustering_score': 0.0,
            'clustered_types': [],
            'focal_types': [],
            'total_focal_types': 0
        }
    
    # Create grid
    x_min, x_max = coord_df['x'].min(), coord_df['x'].max()
    y_min, y_max = coord_df['y'].min(), coord_df['y'].max()
    
    x_bins = np.linspace(x_min, x_max, grid_size[0] + 1)
    y_bins = np.linspace(y_min, y_max, grid_size[1] + 1)
    
    coord_df_temp = coord_df.copy()
    coord_df_temp['grid_x'] = pd.cut(coord_df_temp['x'], x_bins, labels=False, include_lowest=True)
    coord_df_temp['grid_y'] = pd.cut(coord_df_temp['y'], y_bins, labels=False, include_lowest=True)
    
    clustered_types = []
    
    
    for cell_type in focal_types:
        
        # Calculate spatial distribution significance (Chi-square test)
        grid_counts = np.zeros((grid_size[0], grid_size[1]))
        grid_totals = np.zeros((grid_size[0], grid_size[1]))
        
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                grid_cells = coord_df_temp[
                    (coord_df_temp['grid_x'] == i) & 
                    (coord_df_temp['grid_y'] == j)
                ]
                
                type_count = len(grid_cells[grid_cells['cell_type'] == cell_type])
                total_count = len(grid_cells)
                
                grid_counts[i, j] = type_count
                grid_totals[i, j] = total_count
        
        # Chi-square test
        observed = grid_counts.flatten()
        total_per_grid = grid_totals.flatten()
        valid_mask = total_per_grid > 0
        observed_valid = observed[valid_mask]
        total_valid = total_per_grid[valid_mask]
        
        if len(observed_valid) > 1:
            global_prop = np.sum(observed_valid) / np.sum(total_valid)
            expected_valid = total_valid * global_prop
            
            chi2_stat = np.sum((observed_valid - expected_valid)**2 / (expected_valid + 1e-8))
            dof = len(observed_valid) - 1
            spatial_p_value = 1 - chi2.cdf(chi2_stat, dof)
            
            # Calculate Moran's I (spatial autocorrelation)
            grid_centers = []
            grid_props = []
            
            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    if grid_totals[i, j] > 0:
                        center_x = (x_bins[i] + x_bins[i+1]) / 2
                        center_y = (y_bins[j] + y_bins[j+1]) / 2
                        prop = grid_counts[i, j] / grid_totals[i, j]
                        
                        grid_centers.append([center_x, center_y])
                        grid_props.append(prop)
            
            if len(grid_centers) >= 4:
                moran_i = calculate_morans_i(np.array(grid_centers), np.array(grid_props))
            else:
                moran_i = 0
            
            # Clustering criteria
            is_spatially_significant = spatial_p_value < spatial_significance_threshold
            is_spatially_autocorrelated = moran_i > autocorrelation_threshold
            is_clustered = is_spatially_significant and is_spatially_autocorrelated
            
            if is_clustered:
                clustered_types.append(cell_type)
            

            

    
    # Calculate final score (0.0 - 0.5 range)
    n_clustered = len(clustered_types)
    total_focal_types = len(focal_types)
    
    # Score = (clustered_types / focal_types) * 0.5
    clustering_score = (n_clustered / total_focal_types) * 0.5
    

    
    return {
        'clustering_score': clustering_score,
        'clustered_types': clustered_types,
        'focal_types': focal_types,
        'total_focal_types': total_focal_types,
        'n_clustered': n_clustered
    }

def calculate_morans_i(coordinates, values):
    """
    Calculate Moran's I spatial autocorrelation
    """
    n = len(coordinates)
    if n < 4:
        return 0
    
    distances = squareform(pdist(coordinates))
    weights = 1.0 / (distances + 1e-8)
    np.fill_diagonal(weights, 0)
    
    row_sums = weights.sum(axis=1)
    row_sums[row_sums == 0] = 1
    weights = weights / row_sums[:, np.newaxis]
    
    mean_val = np.mean(values)
    centered_values = values - mean_val
    
    if np.sum(centered_values**2) == 0:
        return 0
    
    numerator = np.sum(weights * np.outer(centered_values, centered_values))
    denominator = np.sum(centered_values**2)
    
    return numerator / denominator

def compute_alpha(coord_df):
    """
    Main function: Comprehensive spatial clustering analysis
    Returns clustering score in range 0.0 - 0.5
    """
    
    # Step 1: Identify spatially biased cell types (exclude ubiquitous ones)
    ubiquitous_types, focal_types = identify_spatially_biased_cell_types(coord_df)
    
    # Step 2: Calculate clustering score for focal types only
    score_results = calculate_clustering_score(coord_df, focal_types)
    
    
    return score_results['clustering_score']