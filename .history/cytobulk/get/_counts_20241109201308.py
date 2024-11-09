import pandas as pd
import numpy as np


def get_count_data(adata,counts_location=None):

    data = adata.layers[counts_location].copy() if counts_location else adata.X.copy()
    if not isinstance(data, np.ndarray):
        data= data.toarray()
    data_df = pd.DataFrame(data,index=adata.obs_names,columns=adata.var_names).transpose()
    return data_df

'''
def get_count_data(adata,counts_location=None):

    data = adata.layers[counts_location].copy() if counts_location else adata.X.copy()
    if not isinstance(data, np.ndarray):
        data= data.toarray()
    data_df = pd.DataFrame(data,index=adata.obs_names,columns=adata.var_names).transpose()
    return data_df
'''

def get_count_data_t(adata,counts_location=None):

    data = adata.layers[counts_location].copy() if counts_location else adata.X.copy()
    if not isinstance(data, np.ndarray):
        data= data.toarray()
    return pd.DataFrame(data,index=adata.obs_names,columns=adata.var_names)

