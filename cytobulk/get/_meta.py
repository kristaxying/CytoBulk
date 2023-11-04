import pandas as pd
import anndata as ad

def get_meta(
    adata,
    position_key,
):
    """\
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
        return adata.obs
    


