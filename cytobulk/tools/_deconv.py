
import pandas as pd
from . import model
from .. import get
from .. import utils




def bulk_deconv(bulk_adata, 
                presudo_bulk, 
                sc_adata,
                marker_dict,
                annotation_key = "cell_type", 
                counts_location="batch_effected", 
                out_dir="./out_put"):
    """
    Preprocessing on bulk and sc adata, including following steps:
        1. QC on bulk and sc adata.
        2. Get common gene and common cell type.
        3. Get marker gene which is suitable for this dataset.
        4. Normalization and scale.
        5. Stimulation and batch effects.
        6. NNLS to elimate gap between stimulated bulk and sc adata.
        7. transform gene expression value in input data.

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
    different_source: boolean, optional.
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
    **kwargs: parameters in _filter_adata function.
        
    Returns
    -------
    Returns the preprocessed bulk data (adata) , stimualted bulk data and sc data (adata).
    """

    deconv = model.GraphDeconv(mode="training")
    training_data = get.count_data(presudo_bulk,counts_location=counts_location)
    training_fraction = get.meta(presudo_bulk,position_key="obs")
    test_data = get.count_data(bulk_adata,counts_location=counts_location)

    utils.check_path(out_dir)
    
    deconv.train(out_dir=out_dir,
                expression=training_data,
                fraction=training_fraction,
                marker=marker_dict,
                sc_adata = sc_adata
                )
    deconv.fit(
            out_dir=out_dir,
            expression=test_data,
            marker=marker_dict,
            sc_folder=out_dir+'/cell_feature/',
            model_folder=out_dir+'/model'
        )
    
