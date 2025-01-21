import sys
import os
import numpy as np
import pandas as pd
## IF R path isn't set as system path, using followings to set the config.
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from .. import utils
from .. import get
from os.path import exists



def find_marker_giotto(sc_adata,anno_key,out_dir='./',project='',python_path=None):
    """
    find marker gene for each cell type using Giotto package.

    Parameters
    ----------
    raw_sc_adata : anndata.AnnData
        An :class:`~anndata.AnnData` containing the raw expression.
    annotation_key : string, optional
        The `.obs` key where the single cell annotation is stored.: anndata.AnnData.
    out_dir : string, optional
        The path to save the output file.
    project : string, optional
        The prefix of output file.
        
    Returns
    -------
    None

    """
    # save must be true
    save=True
    if python_path is None:
        # get executed python.exe path
        python_path = sys.executable
        print("your python path is: ",python_path)
    print("your python path is: ",python_path)
    # format expression data
    exp = get.count_data(sc_adata)
    sc_anno = pd.DataFrame(sc_adata.obs[anno_key])
    #sc_anno.index.name="cell_ID"
    #sc_anno['cell_ID'] = sc_anno.index
    sc_anno.insert(0, 'cell_ID', sc_anno.index)
    
    # get r script path
    current_file_dir = os.path.dirname(__file__)

    robjects.r.source(current_file_dir+'/cytobulk_preprocessing.R')
    try:
        robjects.r.source(current_file_dir+'/cytobulk_preprocessing.R')
        print("R script loaded successfully.")
    except Exception as e:
        print(f"An error occurred while sourcing the R script: {e}")
    try:
        run_giotto = robjects.r['run_giotto']
        print("run_giotto function is loaded and ready to use.")
    except KeyError:
        print("run_giotto function is not found in the R environment.")
    # auto trans from pandas to r dataframe

    pandas2ri.activate()
    robjects.r.run_giotto(exp,sc_anno,python_path,out_dir,project)
    # stop auto trans from pandas to r dataframe
    pandas2ri.deactivate()




def remove_batch_effect(pseudo_bulk, bulk_adata, out_dir, project='',batch_effect=True):
    """
    Remove batch effect between pseudo_bulk and input bulk data.

    Parameters
    ----------
    pseudo_bulk : anndata.AnnData
        An :class:`~anndata.AnnData` containing the pseudo expression.
    bulk_adata : anndata.AnnData
        An :class:`~anndata.AnnData` containing the input expression.
    out_dir : string, optional
        The path to save the output file.
    project : string, optional
        The prefix of output file.
        
    Returns
    -------
    Returns the expression after removing batch effect.

    """
    out_dir = utils.check_paths(out_dir+'/batch_effect')
    if batch_effect:
        if exists(f'{out_dir}/{project}_batch_effected.txt'):
            print(f'{out_dir}/{project}_batch_effected.txt already exists, skipping batch effect.')
            bulk_data = pd.read_csv(f"{out_dir}/{project}_batch_effected.txt",sep='\t').T
        else:
            
            save=True
            # check path, file will be stored in out_dir+'/batch_effect'
            pseudo_bulk_df = get.count_data(pseudo_bulk)
            input_bulk_df= get.count_data(bulk_adata)

            bulk = pd.concat([pseudo_bulk_df,input_bulk_df], axis=1)

            cells = np.append(pseudo_bulk.obs_names, bulk_adata.obs_names)
            batch = np.append(np.ones((1,len(pseudo_bulk.obs_names))), np.ones((1,len(bulk_adata.obs_names)))+1)
            if save:
                bulk.to_csv(out_dir+f"/{project}_before_batch_effected.txt",sep='\t')
            meta = pd.DataFrame({"batch": batch,"cells":cells})
            # get r script path
            current_file_dir = os.path.dirname(__file__)
            robjects.r.source(current_file_dir+'/cytobulk_preprocessing.R')
            pandas2ri.activate()
            robjects.r.run_combat(bulk, meta, out_dir, project)
            # stop auto trans from pandas to r dataframe
            pandas2ri.deactivate()
            # add layer
            if exists(f'{out_dir}/{project}_batch_effected.txt'):
                bulk_data = pd.read_csv(f"{out_dir}/{project}_batch_effected.txt",sep='\t').T
                print(bulk_data.shape)
            else:
                raise ValueError('The batch_effected data is not available')
        bulk_data.clip(lower=0,inplace=True)
        pseudo_bulk.layers["batch_effected"] = bulk_data.loc[pseudo_bulk.obs_names,:].values
        bulk_adata.layers["batch_effected"] = bulk_data.loc[bulk_adata.obs_names,:].values
    else:
        pseudo_bulk_df = get.count_data(pseudo_bulk)
        input_bulk_df= get.count_data(bulk_adata)
        bulk = pd.concat([pseudo_bulk_df,input_bulk_df], axis=1)
        bulk.to_csv(out_dir+f"/{project}_batch_effected.txt",sep='\t')

    return pseudo_bulk,bulk_adata
