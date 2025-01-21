
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import matplotlib.patches
import matplotlib.cm as cm
import scanpy as sc
from .. import utils
from .. import get

class Const:
    """
    Some COLOR SET used in the class.
    
    """
    DIFF_COLOR_2 = ['#011F5B','#F19CBB']
    #DIFF_COLOR_N = ['#005f73','#94d2bd','#e9d8a6','#fac748','#e76f51','#9b2226']
    DIFF_COLOR_N = ['#191970','#ADD8E6','#005f73','#94d2bd','#e9d8a6','#fac748','#e76f51','#9b2226']




def _plot_scatter_2label(x1,x2,X1_label,X2_label,title,color_set=None,fig_format=None,out_dir='/out'):
    columns=x1.columns
    plt.figure(figsize=(4, 4))
    if color_set is None:
        color_set = Const.DIFF_COLOR_2
    if fig_format is None:
        fig_format = 'svg'
    s = [0,1]
    data = [x1,x2]
    marker1 = ["^", "o"]
    for index in range(2):
        XOffset = data[index][columns[0]]
        YOffset = data[index][columns[1]]
        s[index] = plt.scatter(XOffset, YOffset, c=color_set[index], s=50, alpha=1, marker=marker1[index], linewidth=0,zorder=-index) 

    plt.legend((s[0],s[1]),(X1_label,X2_label) ,loc = 'best')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(f'Scatter Plot of {title}',fontsize=10) 
    plt.savefig(f'{out_dir}/scatter_{title}.{fig_format}', bbox_inches='tight',transparent=True)


def plot_batch_effect(bulk_adata,pseudo_bulk,out_dir,title=''):
    def _get_paired_data(adata,status=None):
        data = get.count_data(adata,counts_location=status)
        return(utils.pca(data))
    def _plot_paired_data(bulk_adata,pseudo_bulk,title,out_dir,status=None):
        X1 = _get_paired_data(bulk_adata,status)
        X2 = _get_paired_data(pseudo_bulk,status)
        _plot_scatter_2label(X1,X2,X1_label="input_data",X2_label="simulated data",title=title,out_dir=out_dir)

    _plot_paired_data(bulk_adata,pseudo_bulk,title+"(original)",out_dir=out_dir)
    _plot_paired_data(bulk_adata,pseudo_bulk,title+"(batch effected)",out_dir=out_dir,status="batch_effected")        

def plot_celltype_fraction_pie(adata,
                               scale_facter_x=1,
                               scale_factor_y=1,
                               r=0.1,
                               out_dir='/plot',
                               project_name='project',
                               color_list=None,
                               color_map="Spectral",
                               rotation_angle=None,
                               figsize=(2, 2)):
    """
    Plot cell type fraction as pie charts at spatial coordinates.

    This function generates pie charts representing the cell type fraction at each spatial location (from the 'spatial' data in `adata`) and plots them on a 2D scatter plot. Each pie chart is positioned based on the spatial coordinates and reflects the relative abundance of each cell type at that spot. The resulting figure is saved as an SVG file.

    Parameters
    ----------
    adata : anndata.AnnData
        An :class:`~anndata.AnnData` object containing spatial transcriptomics data.
        The spatial coordinates should be stored in `adata.obsm['spatial']`, and the deconvolution results (cell type fractions) should be stored in `adata.uns['deconv']`.
    
    scale_facter_x : float, optional (default: 1)
        A scaling factor to apply to the X-axis coordinates of the spatial data.
    
    scale_factor_y : float, optional (default: 1)
        A scaling factor to apply to the Y-axis coordinates of the spatial data.
    
    r : float, optional (default: 0.1)
        Radius of the pie charts representing cell type fractions at each spatial coordinate.
    
    out_dir : string, optional (default: '/plot')
        The directory where the output figure (SVG format) will be saved.
    
    project_name : string, optional (default: 'project')
        The prefix to use for the saved SVG file.
    
    color_list : list, optional (default: None)
        A list of colors to use for the cell types. If `None`, a colormap (specified by `color_map`) will be used to generate colors for each cell type.
    
    color_map : string, optional (default: 'Spectral')
        The name of the colormap to use if `color_list` is not provided. It will generate a set of colors for the different cell types.
    
    rotation_angle : float, optional (default: None)
        An optional angle (in degrees) to rotate the spatial coordinates. If provided, the rotation is applied to the coordinates before plotting.
    
    figsize : tuple, optional (default: (2, 2))
        The size of the output figure, specified as a tuple of (width, height) in inches.
    
    Returns
    -------
    None
        The function does not return any values but saves the generated pie chart plot as an SVG file in the specified `out_dir`.
    """
    loc_xy=adata.obsm['spatial']
    loc_xy=pd.DataFrame(loc_xy,columns=['x','y'],index=adata.obs_names)
    if rotation_angle:
        new_loc = loc_xy.apply(lambda x : utils.rotate_matrix(x['x'],x['y'],rotation_angle), axis=1)
        loc_xy = pd.DataFrame(list(new_loc),columns=loc_xy.columns,index=loc_xy.index)
    cell_type_fraction = adata.uns['deconv']
    cell_type_fraction['None']=0
    cell_type_fraction.loc[(cell_type_fraction==0).all(axis=1),"None"]=1
    cell_type_fraction['x']=loc_xy['x']
    cell_type_fraction['y']=loc_xy['y']
    loc=cell_type_fraction
    loc['max_idx'] = loc.iloc[:,:-2].idxmax(axis=1)
    loc['x']=loc['x']/scale_facter_x
    loc['y']=loc['y']/scale_factor_y
    fig, ax = plt.subplots(figsize=figsize)
    def _plot_pie(x, ax, color_sets,r): 
        fraction = x[:-3]
        ax.pie(fraction, center=(x['x'],x['y']), radius=r,colors=[color_sets[key] for key in loc.columns[:-3]])
        ax.set_title('CytoBulk',fontsize=20,x=1.2,y=1.7)
    # git min/max values for the axes
    cells = loc.columns[:-3].tolist()
    if color_list:
        color_dict = {item: color for item, color in zip(cells, color_list)}
    else:
        color_map = cm.get_cmap(color_map)
        list_length = len(cells)
        colors_list = [to_hex(color_map(i / (list_length - 1))) for i in range(list_length)]
        color_dict = {item: color for item, color in zip(cells, colors_list)}
    
    loc.apply(lambda x : _plot_pie(x, ax,color_dict,r=0.1), axis=1)
    
    handles = []
    for i, l in enumerate(cells):
        handles.append(matplotlib.patches.Patch(color=colors_list[i], label=cells[i]))
    plt.legend(handles,cells, bbox_to_anchor=(0.1,0.1),fontsize=12,markerscale=0.5,ncol=3,loc="upper left")
    fig.show()
    fig.savefig(f"{out_dir}/{project_name}_cell_fraction_pie.svg", format="svg", dpi=1200,bbox_inches = 'tight',transparent = True)


def plot_celltype_fraction_heatmap(adata,
                                   label,
                                    r=0.1,
                                    out_dir='/plot',
                                    project_name='project',
                                    color_map='crest',
                                    rotation_angle=None,
                                    figsize=(2.7, 2)):
    """
    Plot a heatmap of cell type fractions at spatial coordinates.

    This function visualizes the fraction of a specified cell type as a heatmap on a 2D scatter plot, using spatial transcriptomics data. The color intensity of each point corresponds to the cell type fraction at that spatial location. The resulting figure is saved as an SVG file.

    Parameters
    ----------
    adata : anndata.AnnData
        An :class:`~anndata.AnnData` object containing spatial transcriptomics data.
        The spatial coordinates should be stored in `adata.obsm['spatial']`, and the deconvolution results (cell type fractions) should be stored in `adata.uns['deconv']`.
    
    label : string
        The name of the cell type to plot. This should correspond to one of the columns in `adata.uns['deconv']`, which represents the fractions of different cell types.
    
    r : float, optional (default: 0.1)
        Radius of the points representing the cell type fractions at each spatial coordinate.
    
    out_dir : string, optional (default: '/plot')
        The directory where the output figure (SVG format) will be saved.
    
    project_name : string, optional (default: 'project')
        The prefix to use for the saved SVG file.
    
    color_map : string, optional (default: 'crest')
        The name of the colormap to use for the heatmap. This can be any colormap recognized by `seaborn` or `matplotlib`.
    
    rotation_angle : float, optional (default: None)
        An optional angle (in degrees) to rotate the spatial coordinates. If provided, the rotation is applied to the coordinates before plotting.
    
    figsize : tuple, optional (default: (2.7, 2))
        The size of the output figure, specified as a tuple of (width, height) in inches.
    
    Returns
    -------
    None
        The function does not return any values but saves the generated heatmap plot as an SVG file in the specified `out_dir`.
    """
    from sklearn.preprocessing import minmax_scale
    fig, ax = plt.subplots(figsize=figsize)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    loc_xy=adata.obsm['spatial']
    loc_xy=pd.DataFrame(loc_xy,columns=['x','y'],index=adata.obs_names)
    if rotation_angle:
        new_loc = loc_xy.apply(lambda x : utils.rotate_matrix(x['x'],x['y'],rotation_angle), axis=1)
        loc_xy = pd.DataFrame(list(new_loc),columns=loc_xy.columns,index=loc_xy.index)
    cell_type_fraction = adata.uns['deconv']
    cell_type_fraction['x']=loc_xy['x']
    cell_type_fraction['y']=loc_xy['y']
    cell_type_fraction[label] = minmax_scale(cell_type_fraction[label].values)
    sns.scatterplot(data=cell_type_fraction, x="x", y="y",palette=color_map,hue=label,s=20,legend=False,edgecolor="None",markers="pentagon")
    norm = plt.Normalize(cell_type_fraction[label].min(), cell_type_fraction[label].max())
    cmap = sns.color_palette(color_map, as_cmap=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([ax.get_position().x1+0.05, ax.get_position().y0, 0.03, ax.get_position().height])
    ax.set_title(f'{label}',fontsize=10,x=ax.get_position().x1/2,y=ax.get_position().y1+0.1)
    ax.figure.colorbar(sm, cax=cax)
    fig.show()
    fig.savefig(f"{out_dir}/{project_name}_{label}.svg", format="svg", dpi=1200,bbox_inches = 'tight',transparent = True)

    
def plot_paired_violin(adata,
                       label,
                       gene,
                       stats_method='spearmanr',
                       out_dir='/plot',
                       color_list=Const.DIFF_COLOR_N,
                       project_name="test",
                       figsize=(6, 4),
                       ylim=[-0.1, 1.2]):
    """
    Plot paired violin plots for gene expression and predicted cell type fractions.

    This function generates paired violin plots comparing the expression of a specific gene with the predicted cell type fraction (from deconvolution data) for the same set of observations. It also computes a statistical correlation (Spearman or Pearson) between the gene expression and the predicted cell type fractions and displays the correlation coefficient and significance on the plot.

    Parameters
    ----------
    adata : anndata.AnnData
        An :class:`~anndata.AnnData` object containing gene expression data and the output of a cell type deconvolution.
        - Gene expression data should be stored in `adata.X` with cell/barcode names in `adata.obs_names` and gene names in `adata.var_names`.
        - Predicted cell type fractions (from deconvolution) should be stored in `adata.uns['deconv']`.
    
    label : string
        The name of the cell type (as stored in `adata.uns['deconv']`) to be compared against the gene expression.
    
    gene : string
        The name of the gene (as stored in `adata.var_names`) whose expression levels will be compared to the predicted cell type fractions.
    
    stats_method : string, optional (default: 'spearmanr')
        The statistical method used to compute the correlation between gene expression and predicted cell type fractions.
        - `'spearmanr'`: Spearman’s rank correlation.
        - `'pearsonr'`: Pearson’s correlation.
    
    out_dir : string, optional (default: '/plot')
        The directory where the output violin plot (SVG format) will be saved.
    
    color_list : list, optional (default: Const.DIFF_COLOR_N)
        A list of colors to use for the violin plots. If not provided, a default color palette will be used.
    
    project_name : string, optional (default: 'test')
        The prefix to use for the saved SVG file.
    
    figsize : tuple, optional (default: (6, 4))
        The size of the output figure, specified as a tuple of (width, height) in inches.
    
    ylim : list, optional (default: [-0.1, 1.2])
        The limits for the Y-axis of the violin plot. This should be a list of two values [min, max].
    
    Returns
    -------
    None
        The function does not return any values but saves the generated violin plot as an SVG file in the specified `out_dir`.
    """
    from scipy import stats
    from sklearn.preprocessing import minmax_scale
    fig = plt.figure(figsize=figsize)
    color_panel = sns.set_palette(color_list)
    name_list=[]
    value_list=[]
    exp_data = pd.DataFrame(adata.X,index=adata.obs_names,columns=adata.var_names)
    predicted = adata.uns['deconv']
    common_label=exp_data.index.intersection(predicted.index)
    predicted=predicted.loc[common_label,:]
    exp_data=exp_data.loc[common_label,:]
    exp_data[gene] = minmax_scale((exp_data[gene]), feature_range=(0, 1))
    predicted[label] = minmax_scale(predicted[label], feature_range=(0, 1))
    if stats_method=='spearmanr':
        stat,p_value = stats.spearmanr(exp_data[gene],predicted[label])
    else:
        stat,p_value = stats.pearsonr(exp_data[gene],predicted[label])
    p_value=utils.convert_pvalue_to_asterisks(p_value)
    value_list.extend(exp_data[gene].values)
    name_list.extend([gene]*len(exp_data[gene].values))
    value_list.extend(predicted[label])
    name_list.extend([label]*len(predicted[label]))
    input_data=pd.DataFrame.from_dict({'source':name_list,'value':value_list})
    ax = sns.violinplot(x="source",y="value",data=input_data,hue='source',palette=color_panel)
    x1, x2 = 0, 1
    y,h = predicted[label].max()+.1,.1
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c="k") 
    ax.text((x1+x2)*.5, y+h, stats_method+": "+ str("%.2f" % stat) +" "+  p_value, ha='center', va='bottom', color="k",fontsize=14)
    ax.set_ylim([-0.3, 1.5])
    ax.tick_params(which='major',direction='in',length=3,width=1.,labelsize=14,bottom=False)
    plt.savefig(f"{out_dir}/{project_name}_{label}_{gene}_violin.svg", format="svg", dpi=1200,bbox_inches = 'tight',transparent = True)



def plot_reconstruction(adata,
                        out_dir,
                        project_name="test",
                        rotation_angle=None,
                        spot_size=0.5):
    """
    Plot reconstructed spatial transcriptomics data with correlation analysis.

    This function computes the Pearson correlation between the original and reconstructed spatial transcriptomics data for each spot. It then visualizes the spatial distribution of correlation coefficients using a scatter plot.

    Parameters
    ----------
    adata : anndata.AnnData
        An :class:`~anndata.AnnData` object containing the spatial transcriptomics data.
        - Original data should be in `adata.layers['original_st']`.
        - Reconstructed data should be in `adata.X`.
        - Spatial coordinates should be in `adata.obsm['spatial']`.

    out_dir : string
        The directory where the output plot (SVG format) will be saved.

    project_name : string, optional (default: 'test')
        The prefix to use for the saved SVG file.

    rotation_angle : float, optional
        The angle to rotate the spatial coordinates, if any.

    spot_size : float, optional (default: 0.5)
        The size of the spots in the scatter plot.

    Returns
    -------
    None
        The function does not return any values but saves the generated scatter plot as an SVG file in the specified `out_dir`.
    """
    from scipy.stats import pearsonr
    adata.var_names_make_unique()
    # Calculate Pearson correlation and p-value for each sample (row)
    loc_xy=adata.obsm['spatial']
    loc_xy=pd.DataFrame(loc_xy,columns=['x','y'],index=adata.obs_names)
    if rotation_angle:
        new_loc = loc_xy.apply(lambda x : utils.rotate_matrix(x['x'],x['y'],rotation_angle), axis=1)
        loc_xy = pd.DataFrame(list(new_loc),columns=loc_xy.columns,index=loc_xy.index)

    results = []
    for i in range(adata.shape[0]):  # Loop through each sample
        r, p = pearsonr(adata.layers['original_st'][i], adata.X[i])
        results.append((i, r, p))  # Store index, Pearson r, and p-value

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results, columns=['spot', 'Pearson R', 'pvalue'])
    results_df.set_index('spot',inplace=True)
    adata.obsm['spatial']=loc_xy.values
    adata.obs = results_df
    average_corr = round(np.mean(results_df['Pearson R']),3)

    fig=sc.pl.spatial(
        adata,
        color='Pearson R',
        img_key=None,
        alpha=0.8,
        color_map="mako_r",
        size=1.5,
        title=f'Reconstructed ST\nmean Pearson correlation = {average_corr}\ngene number = {len(adata.var_names)}',
        frameon=False,
        spot_size=spot_size,
        outline_width=0,
        return_fig=True
    )
    plt.savefig(f"{out_dir}/{project_name}_reconstructed_correlation.svg", format="svg", dpi=1200,bbox_inches = 'tight',transparent = True)


    



    
def plot_he_cell_type(data,out_dir):
    """
    Visualize and save a scatter plot of cell locations by cell type.

    This function generates a scatter plot to visualize the spatial distribution of different cell types 
    based on their x and y coordinates. The plot is customized with specific colors for predefined cell types 
    and is saved as a PNG file.

    Parameters
    ----------
    data : pandas.DataFrame
        A pandas DataFrame containing the following columns:
        - 'cell_type': Categorical data representing cell types (e.g., "Epithelial Cells").
        - 'x': Numerical values representing the x-coordinates of the cells.
        - 'y': Numerical values representing the y-coordinates of the cells.

    out_dir : str
        The directory where the generated plot image will be saved. The plot will be saved as 
        "cell_type.png" within this directory.

    Returns
    -------
    None
        This function does not return any values. It generates and saves the scatter plot as a PNG file.

    Notes
    -----
    - The plot uses predefined colors for the following cell types:
        * "Epithelial Cells": #A52A2A (brown)
        * "Neutrophils": #0000B8 (blue)
        * "Plasma Cells": #0D98BA (cyan)
        * "Connective Tissue": #FFCC33 (yellow)
        * "Lymphocytes": #B284BE (purple)
      Cell types not listed above are assigned the color "black".
    - The y-axis is inverted to match the typical orientation of spatial data.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    categories = data["cell_type"].unique()
    colors = {
        "Epithelial Cells": "#A52A2A",  
        "Neutrophils": "#0000B8",      
        "Plasma Cells": "#0D98BA",     
        "Connective Tissue": "#FFCC33", 
        "Lymphocytes": "#B284BE"      
    }

    plt.figure(figsize=(4, 4))

    for category in categories:
        subset = data[data["cell_type"] == category]
        plt.scatter(subset["x"], subset["y"], label=category, color=colors.get(category, "black"), s=10, alpha=0.8)
    plt.legend(
        loc="upper right",             # Place the legend in the top-right corner
        title="Cell Types",            # Add a title to the legend
        fontsize=8,                    # Set font size for the legend labels
        title_fontsize=10,             # Set font size for the legend title
        markerscale=1.5                # Scale the legend markers (scatter points)
    )
    
    
    plt.gca().axis("off")
    plt.gca().invert_yaxis()
    save_path = f"{out_dir}/cell_type.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0) 
    plt.show()


def plot_gene_similarity(adata, marker_df, custom_palette="Spectral"):
    """
    Compute and visualize cosine similarity for marker genes across cell types.

    This function calculates cosine similarity for marker genes between the gene expression matrix 
    and a mapping layer in an AnnData object. It generates a swarm plot to visualize the results 
    grouped by cell type.

    Parameters
    ----------
    adata : anndata.AnnData
        An :class:`~anndata.AnnData` object containing gene expression data in `.X` and a mapping layer in `.layers["mapping_ori"]`.

    marker_df : pandas.DataFrame
        DataFrame containing marker gene information. It must include the following columns:
        - 'gene_symbol': Gene symbol names.
        - 'cell_type': Cell type associated with the marker gene.
        - 'pvalue': P-value indicating marker significance.
        - 'score': Score representing the marker strength.

    custom_palette : str or list, optional (default: "Spectral")
        Custom color palette for plotting. Can be a string representing a seaborn palette or a list of colors.

    Returns
    -------
    matplotlib.pyplot.Figure
        A swarm plot showing cosine similarities for marker genes grouped by cell type.
    """
    # Access the gene expression matrix and mapping layer
    import pandas as pd
    import numpy as np
    import scanpy as sc
    from scipy.spatial.distance import cosine
    import matplotlib.pyplot as plt
    import seaborn as sns
    mapping_ori = adata.layers["mapping_ori"]  # Original mapping layer
    adata_x = adata.X  # Main gene expression matrix

    # Align gene symbols between the marker dataframe and the AnnData object
    genes = adata.var.index.tolist()  # List of genes in AnnData
    marker_genes = marker_df['gene_symbol'].tolist()  # List of marker genes
    common_genes = list(set(marker_genes).intersection(genes))  # Find common genes

    # Filter the marker DataFrame to only include the common genes
    marker_df = marker_df[marker_df['gene_symbol'].isin(common_genes)]

    # Initialize a list to store results for all cell types
    all_results = []

    # Group marker genes by cell type
    for cell_type, group in marker_df.groupby("cell_type"):
        print(f"Processing cell_type: {cell_type}")
        
        # Filter marker genes with p-value < 0.05
        filtered_group = group[group['pvalue'] < 0.05]
        
        # If more than 50 genes are available, select the top 50 based on the 'score' column
        top_genes = filtered_group.nlargest(50, "score")

        # Define a helper function to compute cosine similarity
        def compute_cosine_similarity(gene_symbol, mapping_ori, adata_x, var_names):
            """
            Compute cosine similarity for a given gene.

            Parameters:
            ----------
            gene_symbol : str
                The gene symbol to compute similarity for.
            mapping_ori : np.ndarray
                The original mapping layer array.
            adata_x : np.ndarray
                Gene expression matrix from AnnData.
            var_names : list
                List of gene names in AnnData.

            Returns:
            -------
            float
                Cosine similarity value or NaN if computation is not possible.
            """
            # If the gene is not found in AnnData, return NaN
            if gene_symbol not in var_names:
                return np.nan
            
            # Get the index of the gene in the AnnData object
            gene_idx = var_names.index(gene_symbol)
            
            # Extract expression vectors for the gene
            gene_expression_vector = adata_x[:, gene_idx].flatten()  # Expression vector in adata.X
            ori_vector = mapping_ori[:, gene_idx].flatten()  # Expression vector in mapping_ori

            # Compute cosine similarity if vectors are non-zero
            if np.any(gene_expression_vector) and np.any(ori_vector):
                similarity = 1 - cosine(gene_expression_vector, ori_vector)
            else:
                similarity = np.nan  # Return NaN for zero vectors
            
            return similarity

        # Compute cosine similarity for each gene in the top genes
        var_names = adata.var.index.tolist()  # Get gene names from AnnData
        cosine_similarities = [
            compute_cosine_similarity(gene_symbol, mapping_ori, adata.X, var_names)
            for gene_symbol in top_genes["gene_symbol"]
        ]
        
        # Add cosine similarity values to the DataFrame
        top_genes["cosine_similarity"] = cosine_similarities
        
        # Append the results for the current cell type
        all_results.append(top_genes)

    # Concatenate results from all cell types into a single DataFrame
    final_results = pd.concat(all_results, ignore_index=True)

    # Use seaborn's default palette if no custom palette is provided
  # Use seaborn's default color palette

    # Create a Swarm Plot for cosine similarity grouped by cell type
    plt.figure(figsize=(6, 4))  # Set the figure size to a smaller size (6x4 inches)
    sns.swarmplot(
        x="cell_type", 
        y="cosine_similarity", 
        data=final_results, 
        palette=custom_palette  # Use custom or default palette
    )

    # Add labels and customize the plot
    plt.xlabel("Cell Type", fontsize=12)
    plt.ylabel("Cosine Similarity", fontsize=12)
    plt.xticks(fontsize=10)  # Adjust font size for x-axis labels
    plt.tight_layout()  # Ensure the layout fits well

    # Return the matplotlib.pyplot object for further customization or saving
    return plt