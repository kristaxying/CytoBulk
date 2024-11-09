import numpy as np
import pandas as pd
import math
from scipy.sparse import issparse
from sklearn.preprocessing import StandardScaler
import anndata._core.views
from sklearn.decomposition import PCA
import scanpy as sc
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
    tmp = np.corrcoef(A, B)
    return tmp[0,1] 

def calculate_distance(matrix1,matrix2):
    return (1 - sp.distance.cdist(matrix1, matrix2, 'cosine'))


# Rotation matrix function
def rotate_matrix (x, y, angle, x_shift=0, y_shift=0, units="DEGREES"):
    """
    Rotates a point in the xy-plane counterclockwise through an angle about the origin
    https://en.wikipedia.org/wiki/Rotation_matrix
    :param x: x coordinate
    :param y: y coordinate
    :param x_shift: x-axis shift from origin (0, 0)
    :param y_shift: y-axis shift from origin (0, 0)
    :param angle: The rotation angle in degrees
    :param units: DEGREES (default) or RADIANS
    :return: Tuple of rotated x and y
    """

    # Shift to origin (0,0)
    x = x - x_shift
    y = y - y_shift

    # Convert degrees to radians
    if units == "DEGREES":
        angle = math.radians(angle)

    # Rotation matrix multiplication to get rotated x & y
    yr = (x * math.cos(angle)) - (y * math.sin(angle)) + x_shift
    xr = (x * math.sin(angle)) + (y * math.cos(angle)) + y_shift

    return xr,yr

def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"