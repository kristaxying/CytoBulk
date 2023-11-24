
import pandas as pd
import numpy as np
import os
from numpy.random import choice
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb, to_rgba
from matplotlib import colors
import numpy as np
import pandas as pd
import scanpy as sc
from .. import utils
import matplotlib.patches as mpatches

class Const:
    """
    Some COLOR SET used in the class.
    
    """
    DIFF_COLOR_2 = ['#F9B9B7','#61A0AF']
    #DIFF_COLOR_N = ['#005f73','#94d2bd','#e9d8a6','#fac748','#e76f51','#9b2226']
    DIFF_COLOR_N = ['#191970','#ADD8E6']
    FIG_FORMAT = 'svg'



def plot_scatter_2label(x1,x2,X1_label,X2_label,title,color_set=None,fig_format=None,out_dir='/out'):

    plt.figure(figsize=(4, 4))
    if color_set is None:
        color_set = Const.DIFF_COLOR_2
    if fig_format is None:
        fig_format = Const.FIG_FORMAT
    s = [0,1]
    data = [x1,x2]
    marker1 = ["^", "o"]
    for index in range(2):
        XOffset = data[index]['x']
        YOffset = data[index]['y']
        s[index] = plt.scatter(XOffset, YOffset, c=color_set[index], s=50, alpha=1, marker=marker1[index], linewidth=0,zorder=-index) 

    plt.legend((s[0],s[1]),(X1_label,X2_label) ,loc = 'best')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(f'Scatter Plot of {title}') 
    plt.savefig(f'{out_dir}/scatter_{title}.{fig_format}', bbox_inches='tight')
