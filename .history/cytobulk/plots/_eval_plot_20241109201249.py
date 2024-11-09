
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
#import scanpy as sc
from .. import utils
import matplotlib.patches as mpatches


class Const:
    """
    Some COLOR SET used in the class.
    
    """
    COLOR_BLUE = ['#191970','#ADD8E6','#22AAA1']
    DIFF_COLOR_GRAY = ['#474350','#FAFAC6']
    FIG_FORMAT = 'svg'



def plot_mse_box(ground_truth,df_list,name_list,title,num_column=6,skipping_mse_calculation=False,
                color_set=Const.COLOR_BLUE,fig_format=None,out_dir='/out',filter_list=None,save=True):
    
    sns.set(font_scale=0.8,style="white")
    if skipping_mse_calculation:
        mse_result_list= df_list
    else:
        mse_result_list=[]
        for method_index in range(len(name_list)):
            df = df_list[method_index]
            mse_result = utils.eval_each_sample_mse(ground_truth,df,name_list[method_index])
            mse_result_list.append(mse_result)
    mse_df = pd.concat(mse_result_list)
    cell_name = ground_truth.columns.tolist()
    n_rows = int(len(cell_name)/num_column)+1
    fig, axes = plt.subplots(n_rows, num_column, sharex=True,sharey=False, figsize=(8,20))
    fig.suptitle(title,y=0.9)
    color_panel = sns.set_palette(color_set)
    max_y=[]
    if filter_list is None:
        filter_list = [2]*n_rows
    for i in range(n_rows):
        for j in range(num_column):
            if i*num_column+j<len(cell_name):
                sub_data = mse_df.loc[mse_df['cell_type'] == cell_name[i*num_column+j],"mse"]
                q1 = np.percentile(sub_data, 25)            
                q3 = np.percentile(sub_data, 80)            
                iqr = q3 - q1
                sub_data = sub_data[sub_data<q3+filter_list[i]*iqr]
                max_y.append(max(sub_data))
    for i in range(n_rows):
        each_row_y = max(max_y[i*num_column:(i+1)*num_column-1])
        for j in range(num_column):
            if i*num_column+j<len(cell_name):
                sub_data = mse_df.loc[mse_df['cell_type'] == cell_name[i*num_column+j],:]
                sns.boxplot(data=sub_data, x='method', y='mse',palette=color_panel,showfliers=False,width=.5,ax=axes[i, j])
                # rename for special cells
                if cell_name[i*num_column+j]=="T CD8 recently activated effector memory":
                    axes[i, j].set_title("T CD8 recently activated EM cells")
                if cell_name[i*num_column+j]=="pDC":
                    axes[i, j].set_title("pDC cells")
                else:
                    axes[i, j].set_title(cell_name[i*num_column+j])
                axes[i, j].set_xlabel("")
                axes[i, j].set_ylabel("")
                # reset the ylim
                if i==8:
                    axes[i, j].set_ylim(0,0.075)
                else:
                    axes[i, j].set_ylim(0,each_row_y)
                if j==0:
                    axes[i, j].set_yticklabels(axes[i, j].get_yticklabels())
                else:
                    axes[i, j].set_yticklabels("")
            else:
                custom_lines = []
                axes[i, j].set_xlabel("")
                axes[i, j].set_ylabel("")
                for m_index in method_index:
                    tmp_legand = mpatches.Rectangle([0,0],color=color_set[m_index],height=20,width=20,label=name_list[m_index],lw=0.5,edgecolor="black")
                    custom_lines.append(tmp_legand)
                axes[i,j].legend(handles=custom_lines, loc='upper center',bbox_to_anchor=(0.5,0.7),edgecolor='w',ncols=1)
                axes[i,j].axis('off')
    plt.edgecolor='white'
    if save:
        plt.savefig(f'{out_dir}/MSE_plot.{fig_format}', bbox_inches='tight')
