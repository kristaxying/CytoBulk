#from cellpose import models, io, plot
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys


class Const:
    PIX_X = 'X'
    PIX_Y = 'Y'

def rgb2grey(img: np.ndarray):
    return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

def predict_cell_num(st_adata_path,
                     diameter,
                     save_png_result=False,
                     model_type='cyto3',
                     out_path='./',
                     cellprob_threshold=1,
                     save=True):
    
    '''
    description.
    Params
    -----
    st_adata_path : string 
        the path of ST adata.
    crop_r : int 
        the length of the tile. each tile will be a square.
    save_png_result : None or string
        if None, do not save the segmentation result to a png file; otherwise, save the png to the provided file path name.

    Returns
    -----
    '''
    print('-----Initializing model...')
    model = models.Cellpose(model_type=model_type)
    ch = [0, 0] # NOTE: here we set all images to greyscale
    st_adata = sc.read_h5ad(st_adata_path)

    print('-----Reading files...')
    img = rgb2grey(st_adata.uns['spatial']['CytAssist_FFPE_Human_Breast_Cancer']['images']['hires'])
    print(img.shape)

    coord = st_adata.obsm['spatial']*st_adata.uns['spatial']['CytAssist_FFPE_Human_Breast_Cancer']['scalefactors']['tissue_hires_scalef']
    spots = pd.DataFrame(coord, columns=["X", "Y"])
    crop_r = int(st_adata.uns['spatial']['CytAssist_FFPE_Human_Breast_Cancer']['scalefactors']['fiducial_diameter_fullres']*st_adata.uns['spatial']['CytAssist_FFPE_Human_Breast_Cancer']['scalefactors']['tissue_hires_scalef'])
    half_r = crop_r // 2 + 2
    print(half_r)
    print('-----Predicting cell number...')
    ret = pd.DataFrame(data={'X':[], 'Y':[], 'cell_num':[]})
    cell_pos = pd.DataFrame(data={'id':[], 'X':[], 'Y':[]})
    for _, row in tqdm(spots.iterrows()):
        x = int(row[Const.PIX_X]); y = int(row[Const.PIX_Y])
        x_max = min(x+half_r, img.shape[0]-1)
        x_min = max(x-half_r, 0)
        y_max = min(y+half_r, img.shape[1]-1)
        y_min = max(y-half_r, 0)

        tile = img[x_min:x_max, y_min:y_max]
        masks, flows, styles, diams = model.eval(tile, diameter=diameter, channels=ch,  cellprob_threshold=cellprob_threshold)
        cell_num = len(np.unique(masks))
        ret.loc[len(ret.index)] = [x, y, cell_num]
        for i in range(cell_num):
            xi = np.where(masks == i)[0].mean()
            yi = np.where(masks == i)[1].mean()
            cell_pos.loc[len(cell_pos.index)] = [f"spot{_}_cell{i}", xi, yi]
        
        if save_png_result:
            fig = plt.figure()
            plot.show_segmentation(fig, tile, masks, flows[0], channels=ch)
            plt.tight_layout()
            #plt.savefig(save_png_result.replace('.', f'_{x}x{y}.'))
            plt.savefig(f"{out_path}figures/{_}_segmentation_result.png")

    st_adata.obsm["cell_num"] = (ret["cell_num"]).to_numpy()
    st_adata.uns["seg_cell_pos"] = cell_pos
    st_adata.write_h5ad(f"{out_path}segmentation_adata.h5ad")
    return st_adata, cell_pos
