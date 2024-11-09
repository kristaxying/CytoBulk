import pandas as pd

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
    elif position_key=="obsm":
        return pd.DataFrame(adata.obsm[columns],index=adata.obs_names,columns=[columns])
    elif position_key=="uns":
        return adata.uns[columns]
    




def get_coords(visium_adata):
    if all(item in visium_adata.obs.columns for item in ['array_row', 'array_col']):
        df_coords = visium_adata.obs[['array_row', 'array_col']]
        df_coords.columns = ['row','col']
    else:
        df_coords = pd.DataFrame(visium_adata.obsm['spatial'],index=visium_adata.obs_names,columns=['row','col'])

    df_coords.index.name = 'SpotID'

    return df_coords
    
    