import sys
import os
import numpy as np
import pandas as pd
import time
## IF R path isn't set as system path, using followings to set the config.
'''
os.environ["R_HOME"] = "D:/R/R-4.3.1" 
os.environ["PATH"] = "D:/R/R-4.3.1/bin/x64" + ";" + os.environ["R_HOME"] 
'''
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from .. import utils
from .. import get
from os.path import exists



def find_marker_giotto(sc_adata,anno_key,out_dir='./',project=''):
    """
    find marker gene for each cell type using Giotto package.

    Parameters
    ----------
    raw_sc_adata: anndata.AnnData
        An :class:`~anndata.AnnData` containing the raw expression.
    annotation_key: string, optional
        The `.obs` key where the single cell annotation is stored.: anndata.AnnData.
    out_dir: string, optional
        The path to save the output file.
    project: string, optional
        The prefix of output file.
        
    Returns
    -------
    None
    """
    # save must be true
    save=True
    # get executed python.exe path
    python_path = sys.executable
    # format expression data
    exp = get.count_data(sc_adata)
    sc_anno = sc_adata.obs[anno_key]
    # get r script path
    current_file_dir = os.path.dirname(__file__)
    robjects.r.source(current_file_dir+'/cytobulk_preprocessing.R')
    # auto trans from pandas to r dataframe
    pandas2ri.activate()
    robjects.r.run_giotto(exp,sc_anno,python_path,out_dir,project,robjects.vectors.BoolVector([save]))
    # stop auto trans from pandas to r dataframe
    pandas2ri.deactivate()



def remove_batch_effect(pseudo_bulk, bulk_adata, out_dir, project='',save=True):
    """
    Remove batch effect between pseudo_bulk and input bulk data.

    Parameters
    ----------
    pseudo_bulk: anndata.AnnData
        An :class:`~anndata.AnnData` containing the pseudo expression.
    bulk_adata: anndata.AnnData
        An :class:`~anndata.AnnData` containing the input expression.
    out_dir: string, optional
        The path to save the output file.
    project: string, optional
        The prefix of output file.
        
    Returns
    -------
    Returns the expression after removing batch effect.
    """
    # save must be true
    save=True
    # check path, file will be stored in out_dir+'/batch_effect'
    out_dir = utils.check_paths(out_dir+'/batch_effect')
    pseudo_bulk_df = get.count_data(pseudo_bulk)
    input_bulk_df= get.count_data(bulk_adata)
    bulk = pd.concat([pseudo_bulk_df,input_bulk_df], axis=1)
    cells = np.append(pseudo_bulk.obs_names, bulk_adata.obs_names)
    batch = np.append(np.ones((1,len(pseudo_bulk.obs_names))), np.ones((1,len(bulk_adata.obs_names)))+1)
    if save:
        bulk.to_csv(out_dir+f"/{project}_before_batch_effected.txt",sep='\t')
    meta = pd.DataFrame({"batch": batch}, index=cells)
    # get r script path
    current_file_dir = os.path.dirname(__file__)
    robjects.r.source(current_file_dir+'/cytobulk_preprocessing.R')
    pandas2ri.activate()
    result = robjects.r.run_combat(bulk, meta,out_dir, project)
    # stop auto trans from pandas to r dataframe
    pandas2ri.deactivate()
    # add layer
    if exists(f'{out_dir}/{project}_batch_effected.txt'):
        bulk_data = pd.read_csv(f"{out_dir}/{project}_batch_effected.txt",sep='\t').T
    else:
        raise ValueError('The batch_effected data is not available')
    bulk_data.clip(lower=0,inplace=True)
    pseudo_bulk.layers["batch_effected"] = bulk_data.loc[pseudo_bulk.obs_names,:].values
    bulk_adata.layers["batch_effected"] = bulk_data.loc[bulk_adata.obs_names,:].values

    return pseudo_bulk,bulk_adata
