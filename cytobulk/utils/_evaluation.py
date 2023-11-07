
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import pearsonr
import numpy as np
from ._read_data import check_paths

def _p_value(data1,data2):
    pccs = pearsonr(data1, data2)
    pval = pccs.pvalue
    stat = '{:.2f}'.format(pccs.statistic)
    return pval,stat


def eval_fraction(df1,df2,out_dir=',',save=True):
    cells=[]
    sig=[]
    mse=[]
    p_value=[]
    cell_name = df1.columns.values.tolist()
    for j in range(len(cell_name)):
        data1 = df1.loc[:,cell_name[j]].values
        data2 = df2.loc[:,cell_name[j]].values
        mse_tmp = mean_squared_error(data1, data2)
        pval,stat = _p_value(data1,data2)
        cells.append(cell_name[j])
        sig.append(float(stat))
        if pval <=0.001:
            str_pval = '***'
        elif 0.001<pval<=0.01:
            str_pval = '**'
        elif 0.01<pval<=0.05:
            str_pval = '*'
        elif str(pval)=="nan":
            str_pval = 'X'
        p_value.append(str_pval)
        mse.append(mse_tmp)
    dict_data = {"cell type":cells,"person correlation":sig,"p_value":p_value,"mse":mse}
    eval_result = pd.DataFrame(dict_data)
    out_dir = check_paths(out_dir+'/output')
    if save:
        eval_result.to_csv(out_dir+"/prediction_eval.txt",sep='\t')
    return eval_result

