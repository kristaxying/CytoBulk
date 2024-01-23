import pandas as pd
import anndata as ad

def get_meta(
    adata,
    position_key="obs",
    columns=None
):
    """
    Get an :class:`~pandas.DataFrame` with the positions of the observations.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` with positions annotation in `.obs` or
        `.obsm`. Can also be a :class:`~pandas.DataFrame`, which is then
        treated like the `.obs` of an :class:`~anndata.AnnData`.
    position_key
        The `.obsm` key or array-like of `.obs` keys with the position space
        coordinates.
        
    Returns
    -------
    A :class:`~pandas.DataFrame` with the positions of the observations.
    
    """
    
    if position_key=="obs":
        if not columns:
            return adata.obs
        else:
            return adata.obs[columns]




def get_coords(visium_adata):
    df_coords = visium_adata.obs[['array_row', 'array_col']]
    df_coords.columns = ['row','col']
    df_coords.index.name = 'SpotID'

    return df_coords