import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix, csc_matrix
from sklearn.utils.sparsefuncs import inplace_column_scale, inplace_row_scale
from sklearn.preprocessing import StandardScaler
import numba
# import mkl
from numba import njit, prange
import anndata._core.views
from sklearn.decomposition import PCA
import scanpy as sc
from sklearn.metrics.pairwise import cosine_similarity
import scipy.spatial as sp

def get_sum(
    X,
    axis,
    dtype=None,
):

    """
    Calculates the sum of a sparse matrix or array-like in a specified axis and
    returns a flattened result.
    
    Parameters
    ----------
    X
        A 2d :class:`~numpy.ndarray` or a `scipy` sparse matrix.
    axis
        The axis along which to calculate the sum
    dtype
        The dtype used in the accumulators and the result.
        
    Returns
    -------
    The flattened sums as 1d :class:`~numpy.ndarray`.
        
    """

    if issparse(X):
        result = X.sum(axis=axis, dtype=dtype).A.flatten()
    else:
        result = X.sum(axis=axis, dtype=dtype)
    

    if isinstance(result, anndata._core.views.ArrayView):
        result = result.toarray()
    
    return result



@njit(fastmath=True, parallel=True, cache=True)
def _log1p(x):
    for i in prange(len(x)):
        x[i] = np.log1p(x[i])

def log1p(
    X,
):

    """
    Calculates log1p inplace and in parallel.
    
    Parameters
    ----------
    X
        A :class:`~numpy.ndarray` with more than 1 dimension, a `scipy` sparse
        matrix, or something which has an attribute `.X` which fits this
        description, e.g. an :class:`~anndata.AnnData`
        
    Returns
    -------
    `None`. This is an inplace operation.
        
    """

    if hasattr(X, 'X'):
        X = X.X

    if issparse(X):
        _log1p(X.data)
    else:
        _log1p(X)

def pca(X,dimension=2):
    
    """
    Calculates decompositioned data  with 2 dimensions.
    
    Parameters
    ----------
    X
        A :class:`~pd.dataframe` with more than 2 dimension.
    dimension: int
        The number to indicate needed dimension.

        
    Returns
    -------
    The dataframe after dimension reduction with PCA function in sklearn.decomposition.
        
    """

    pca = PCA(n_components=dimension)
    new_data = pca.fit_transform(X.values)

    return pd.DataFrame(
        new_data,
        index=X.index,
        columns=[f'PCA{str(x)}' for x in range(1, dimension + 1)],
    )


def normalization_cpm(adata,scale_factors=None,trans_method=None,layer=None):
    """
    Normalize counts per cell.

    Parameters
    ----------
    scale_factors: int, optional
        After normalization, each observation (cell) has a total count equal to the median 
        of total counts for observations (cells) before normalization.
    trans_method: None or 'log', optional
        If log, Computes X=log(X+1), where log denotes the natural logarithm unless a different base is given.
    layer:
        
    Returns
    -------
    Returns the expression after removing batch effect.

    """
    data = adata.copy()
    if scale_factors is not None:
        sc.pp.normalize_total(data, target_sum=scale_factors)
    if trans_method == 'log':
        sc.pp.log1p(data)
    return data

def normal_center_df(data):
    scaler = StandardScaler()
    scaler.fit(data.values)
    trans_data = scaler.transform(data.values)
    return pd.DataFrame(trans_data,index=data.index,columns=data.columns)

    
def pear(A,B):
    tmp = np.corrcoef(A.flatten(), B.flatten())
    return tmp[0,1] 

def calculate_distance(matrix1,matrix2):
    return (1 - sp.distance.cdist(matrix1, matrix2, 'cosine'))
    