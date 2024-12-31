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

np.random.seed(42)

# Modify the target number of classes and the path to the dataset
#NUM_CLASSES = 41
#PATH_TO_SAMPLE_FOLDER = "data/sample_TCGA_images/"


def process_txt_files(input_dir, output_file_name, project_name_filter=None):
    output_file_name=output_file_name+".txt"
    # 输出文件完整路径
    output_file_path = os.path.join(input_dir, output_file_name)

    # 打开输出文件
    with open(output_file_path, "w") as outfile:
        # 遍历输入文件夹中的所有txt文件
        for filename in os.listdir(input_dir):
            if filename.endswith(".txt"):
                # 按文件名格式解析project_name, x, y
                parts = filename.split("_")
                if len(parts) < 4 or not parts[-1].endswith("cent.txt"):
                    print(f"Skipping invalid file name: {filename}")
                    continue
                
                project_name = "_".join(parts[:-4])  # 合并 project_name
                x_offset = int(parts[-4])
                y_offset = int(parts[-3])
                
                # 如果设置了过滤条件，跳过不匹配的project_name
                if project_name_filter and project_name != project_name_filter:
                    continue

                input_path = os.path.join(input_dir, filename)
                
                # 打开并读取txt文件
                with open(input_path, "r") as infile:
                    for line in infile:
                        line = line.strip()
                        if not line:  # 忽略空行
                            continue
                        
                        # 解析每行内容
                        try:
                            x, y, label = line.split("\t")
                            print(x, y, label)
                            x = int(x) + x_offset
                            y = int(y) + y_offset
                            
                            # 写入到输出文件
                            outfile.write(f"{project_name}\t{x}\t{y}\t{label}\n")
                        except ValueError:
                            print(f"Skipping malformed line in {filename}: {line}")

    print(f"Data successfully written to {output_file_path}")

def inference_cell_type_from_he_image(
        image_dir,
        out_dir,
        project):
    if exists(f'{out_dir}/combinded_cent.txt'):
        df = pd.read_csv(f'{out_dir}/combinded_cent.txt',sep='\t')
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
            df = pd.read_csv(f'{out_dir}/combinded_cent.txt',sep='\t')
            df.columns = ['data_set','x','y','cell_type']
            print("Save file done")
        print(df)
    return df

def load_graph1(cell_data, k=15):
    from scipy.spatial.distance import cdist
    coordinates = cell_data[['x', 'y']].values
    labels = cell_data['cell_type'].values
    distances = cdist(coordinates, coordinates, metric="euclidean")
    adjacency_matrix = np.ones_like(distances)
    for i in range(len(distances)):
        nearest_indices = np.argsort(distances[i])[:k + 1]  # K 个最近邻
        adjacency_matrix[i, nearest_indices] = 0
    return adjacency_matrix, labels

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
    lr_affinity_sum = (abs(lr_affinity_matrix_part1) + abs(lr_affinity_matrix_part2))/2

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
    return distance_matrix, sampled_adata.obs[annotation_key].values,sampled_adata


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



