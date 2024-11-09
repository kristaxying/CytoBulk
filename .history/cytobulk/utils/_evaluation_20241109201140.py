from sklearn.metrics import mean_squared_error
import pandas as pd
from scipy.stats import pearsonr
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
        str_pval='none'
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



def eval_comparsion(df1,df2_list,method_name,out_dir=',',save=True):
    cells=[]
    sig=[]
    mse=[]
    p_value=[]
    cell_name = df1.columns.values.tolist()
    methods = []
    for i in range(len(df2_list)):
        for j in range(len(cell_name)):
            data1 = df2_list[i].loc[:,cell_name[j]].values
            data2 = df1.loc[:,cell_name[j]].values
            mse_tmp = mean_squared_error(data1, data2)
            pval,stat = _p_value(data1,data2)
            cells.append(cell_name[j])
            sig.append(float(stat))
            methods.append(method_name[i])
            if pval <=0.001:
                p_value.append('***')
            elif 0.001<pval<=0.01:
                p_value.append('**')
            elif 0.01<pval<=0.05:
                p_value.append('*')
            elif str(pval)=="nan":
                p_value.append('X')
            else:
                p_value.append('X')
            mse.append(mse_tmp)
    dict_data = {"cell type":cells,"method":methods,"person correlation":sig,"p_value":p_value,"mse":mse}
    eval_result = pd.DataFrame(dict_data)
    out_dir = check_paths(out_dir+'/output')
    if save:
        eval_result.to_csv(out_dir+"/fraction_eval.txt",sep='\t')
    return eval_result


def eval_each_sample_mse(df1,df2,method_name,out_dir=',',save=True):
    cells=[]
    mse=[]
    samples=[]
    cell_name = df1.columns.values.tolist()
    methods = []
    sample_name = df1.index.tolist()
    df2 = df2.loc[sample_name,:]
    for i in range(len(df1)):
        for j in range(len(cell_name)):
            data1 = df2.loc[i,cell_name[j]].values
            data2 = df1.loc[i,cell_name[j]].values
            mse_tmp = mean_squared_error(data1, data2)
            samples.append(sample_name[i])
            cells.append(cell_name[j])
            methods.append(method_name)
            mse.append(mse_tmp)
    dict_data = {"sample_name":samples,"cell_type":cells,"method":methods,"mse":mse}
    eval_result = pd.DataFrame(dict_data)
    if save:
        out_dir = check_paths(out_dir+'/output')
        eval_result.to_csv(out_dir+"/fraction_eval.txt",sep='\t')
    return eval_result