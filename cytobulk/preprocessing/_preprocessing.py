import numpy as np
import pandas as pd
import scanpy as sc
from .. import utils
from .. import get
from .. import plots
from ._rpackage import find_marker_giotto,remove_batch_effect
from ._filtering import qc_bulk_sc
import warnings
from os.path import exists
import json
import sys




def _check_cell_type(sc_adata,marker_data=None,cell_type_key="cell_type",cell_list=None):
    
    """
    check the cell type in single cell whether could be found in marker gene profile.

    Parameters
    ----------
    marker_gene: dataframe
        An :class:`~pandas.dataframe` marker gene in of each cell type.
        Each column is marker genes of one cell type. the first row should be name of cell types.
    sc_adata: ~anndata.AnnData
        An :class:`~anndata.AnnData` containing the sc rna expression to filter.
    cell_type_key: string.
        The `.obs` key where the annotation is stored in sc adata.

    Returns
    -------
    Returns the filtered marker gene list.
    
    """
    sc_cells = np.unique(sc_adata.obs[cell_type_key])
    if marker_data is not None and not isinstance(marker_data, pd.DataFrame):
        marker_cells = marker_data.columns
    else:
        marker_cells = sc_cells

    if cell_list is not None:
        common_cell = np.intersect1d(np.array(cell_list), sc_cells, assume_unique=False, return_indices=False)
    else:
        common_cell = sc_cells
    
    common_marker = np.intersect1d(marker_cells, common_cell, assume_unique=False, return_indices=False)
    
    if len(marker_cells) != len(common_marker): # filter happened
        warnings.warn("In marker gene profile, could not find all the cell type of sc adata. ")
    
    if marker_data is not None:
        marker_gene = marker_data[common_marker].to_dict('list')
    else:
        marker_gene = dict.fromkeys(common_cell)

    return marker_gene,common_cell




def _join_marker(sc_adata,annotation_key,marker_dict,common_cell,out_dir='./',dataset_name=''):
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
        marker_gene = find_marker_giotto(sc_adata,annotation_key,out_dir,dataset_name)
    else:
        print(f'{out_dir}/{dataset_name}_marker.txt already exists, skipping find marker.')
        marker_gene = pd.read_csv(f"{out_dir}/{dataset_name}_marker.txt",sep='\t')
    
    
    return utils.data_dict_integration(marker_gene,marker_dict,common_cell,top_num=100)



def _normalization_data(bulk_adata,sc_adata,scale_factors=None,trans_method='log1p',save=False,project='',out_dir='./'):
    """
    Normalization on bulk and sc adata.
        CPM = readsMappedToGene * 1/totalNumReads * 106
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
        - "log1p": log(x+1)
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

def _plot_pca_scatter(bulk_adata, pseudo_bulk, out_dir):
    out_dir = utils.check_paths(f'{out_dir}/plots')
    # before batch effect
    before_x1 = get.count_data(bulk_adata)
    before_x2 = get.count_data(pseudo_bulk)
    plots.scatter_2label(utils.pca(before_x1),utils.pca(before_x2), X1_label="input bulk", 
                        X2_label="reference bulk", title="before batch effect", out_dir=out_dir)
    # before batch effect
    after_x1 = get.count_data(bulk_adata,counts_location="batch_effected")
    after_x2 = get.count_data(pseudo_bulk,counts_location="batch_effected")
    plots.scatter_2label(utils.pca(after_x1),utils.pca(after_x2),X1_label="input bulk", 
                        X2_label="reference bulk", title="after batch effect", out_dir=out_dir)

def align(bulk_adata, pseudo_bulk,save):

    def _compute_scale_factors(A,B):
        
        return bulk_adata
    
    A = pseudo_bulk.layers["batch_effected"]
    B = pseudo_bulk.layers["cell_average_bulk"]
    scale_factor =  _compute_scale_factors(A,B)
    bulk_adata.layers["transformed"] = scale_factor*bulk_adata.layers["batch_effected"]
    bulk_adata.var.scale_factor = scale_factor
    return bulk_adata


def preprocessing(bulk_data,
                sc_adata,
                annotation_key,
                marker_data=None,
                rename=None,
                dataset_name="",
                out_dir='./',
                different_source=True,
                cell_list=None,
                scale_factors=10000,
                trans_method="log",
                save = True,
                save_figure=True,
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
        - "log1p": log(x+1)
        - `None`: no transformation
    save_figure: boolean, optional
        Whether save figures during preprocessing. eg. scatter plot of pca data.
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
        meta = sc_adata.obs
        meta['curated_cell_type'] = meta.apply(lambda x: rename[x[annotation_key]] if x[annotation_key] in rename else "invalid", axis=1)
        annotation_key = 'curated_cell_type'
        print("Finish renaming, the curated annotation could be found in sc_adata.obs['curated_cell_type']")

    print('================================================================================================================')
    print('Start to check cell type annotation and quality control...')
    sc_adata = sc_adata[sc_adata.obs[annotation_key]!="invalid",:]

    sc_adata, bulk_adata, common_gene = qc_bulk_sc(bulk_data,sc_adata,save=save,**kwargs)

    db_marker,common_cell = _check_cell_type(sc_adata,marker_data,annotation_key,cell_list)
    print('Finish quality control.')

    print('Start to find vaild marker genes')
    if db_marker.values() is not None:
        for i in db_marker.keys():
            db_gene = np.array(db_marker[i])
            tmp = np.intersect1d(db_gene,common_gene)
            if len(db_marker) != len(tmp):
                db_marker[i] = tmp

    marker_dict,marker_gene,common_cell = _join_marker(sc_adata,annotation_key,db_marker,common_cell,out_dir,dataset_name)
    # debug
    print('Finish finding vaild marker genes')
    sc_adata = sc_adata[sc_adata.obs[annotation_key].isin(common_cell),marker_gene]
    bulk_adata = bulk_adata[:,marker_gene]
    print(f'The number of valid single cell is {sc_adata.shape[0]}, valid sample is {bulk_adata.shape[0]},the number of valid genes is {sc_adata.shape[1]}')
    
    if scale_factors is not None or trans_method is not None:
        sc_adata.layers["original"] = sc_adata.X
        bulk_adata.layers["original"] = bulk_adata.X
        sc_adata, bulk_adata = _normalization_data(bulk_adata,sc_adata,scale_factors,trans_method,
                                                save=save,project=dataset_name,out_dir=out_dir)
    print('================================================================================================================')
    
    pseudo_adata = utils.bulk_simulation(sc_adata, 
                    common_cell, 
                    annotation_key = annotation_key,
                    project=dataset_name, 
                    out_dir=out_dir,
                    n_sample_each_group=500,
                    min_cells_each_group=100,
                    cell_gap_each_group=100,
                    group_number=5,
                    save=True,
                    return_adata=True)
    if different_source:
        print('================================================================================================================')
        # get cell type average expression.
        print('Start data integration')
        print('Compute average gene expression over different cell types with sc data')
        #average_cell_exp = utils.compute_cluster_averages(sc_adata,annotation_key,common_cell,
                                                        #out_dir=out_dir,project=dataset_name,save=True)
        print('Done')
        # remove batch effect between stimulated bulk rna and input bulk rna
        print('Start batch effect')
        pseudo_bulk, bulk_adata = remove_batch_effect(pseudo_adata, bulk_adata, out_dir=out_dir, project=dataset_name)
        if save_figure:
            _plot_pca_scatter(bulk_adata, pseudo_bulk,out_dir)
        print('Done')
        #cell_average_bulk = utils.compute_bulk_with_average_exp(pseudo_bulk, average_cell_exp, save=save, out_dir=out_dir, project=dataset_name)
        #pseudo_bulk.layers["cell_average_bulk"] = cell_average_bulk
        if save:
            out_dir = utils.check_paths(f'{out_dir}/filtered')
            pseudo_bulk.write_h5ad(f"{out_dir}/pseudo_bulk_{dataset_name}.h5ad")
            sc_adata.write_h5ad(f"{out_dir}/sc_data_{dataset_name}.h5ad")
            bulk_adata.write_h5ad(f"{out_dir}/bulk_data_{dataset_name}.h5ad")
            with open(f"{out_dir}/marker_dict.json", "w") as outfile: 
                json.dump(marker_dict, outfile)

    return sc_adata, pseudo_bulk, bulk_adata, marker_dict, annotation_key

    
