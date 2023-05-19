"""test
Usage:
    test.py [--st_data=<sn>] [--st_meta=<mn>]  [--sc_data=<cn>]  [--sc_meta=<en>]
    test.py (-h | --help)
    test.py --version

Options:
    -s --st_data=<sn>   St exp data.
    -m --st_meta=<mn>   St meta data.
    -c --sc_data=<sn>   Sc exp data.
    -e --se_meta=<en>   Sc meta data.
    -h --help   Show this screen.
    -v --version    Show version.
"""


import pandas as pd
import numpy as np
from docopt import docopt
from CytoBulk import *
import scanpy as sc
#import cell2location



def main(arguments):
    '''
    st_data = arguments.get("--st_data")
    st_meta = arguments.get("--st_meta")
    sc_data = arguments.get("--sc_data")
    sc_meta = arguments.get("--sc_meta")
    #cell_fraction = arguments.get("--cell_fraction")
    st_data = pd.read_csv(st_data,sep='\t',header=0,index_col = 0)
    st_meta = pd.read_csv(st_meta,sep='\t',header=0, index_col = 0, usecols=['spot_id','x_coord','y_coord','sample'])
    sc_data = pd.read_csv(sc_data,sep='\t',header=0,index_col = 0)
    sc_meta = pd.read_csv(sc_meta,sep='\t',header=0,index_col = 0)
    '''
    '''
    M = CytoBulk()
    M.set_image_exp(st_data)
    M.set_st_meta(st_meta)
    M.set_sc_meta(sc_meta)
    M.set_sc_data(sc_data)
    M.spatial_mapping()
    '''
    sc = './data/CHOL_GSE142784_expression.h5ad'
    #meta = './data/CHOL_GSE142784_meta.txt'
    T = CytoBulk()
    T.bulk_deconv(mode='training',training_sc=sc,marker_label='auto_find')
    







if __name__=="__main__":
    arguments = docopt(__doc__, version="test 1.0.0")
    main(arguments)