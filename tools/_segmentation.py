from cellpose import models, io, plot
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from tqdm import tqdm


class Const:
    PIX_X = 'X'
    PIX_Y = 'Y'

def rgb2grey(img: np.ndarray):
    return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

def predict_cell_num(st_adata_path,
                     crop_r,
                     save_png_result=None,
                     model_type='cyto3', 
                     cellprob_threshold=1):
    '''
    Params
    -----
    st_adata_path
    crop_r: the length of the tile. each tile will be a square.
    save_png_result: None or str, if None, do not save the segmentation result to a png file; otherwise, save the png to the provided file path name.

    Returns
    -----
    '''

    print('-----Initializing model...')
    model = models.Cellpose(model_type=model_type)
    ch = [0, 0] # NOTE: here we set all images to greyscale
    st_adata = sc.read_h5ad(st_adata_path)

    print('-----Reading files...')
    img = rgb2grey(st_adata.uns['spatial']['library_ids']['images']['hires'])
    coord = st_adata.obsm['spatial']*st_adata.uns['spatial']['library_ids']['scalefactors']['tissue_hires_scalef']
    spots = pd.DataFrame(coord, columns=["X", "Y"])
    half_r = crop_r // 2
    
    print('-----Predicting cell number...')
    ret = pd.DataFrame(data={'X':[], 'Y':[], 'cell_num':[]})
    cell_pos = pd.DataFrame(data={'id':[], 'X':[], 'Y':[]})
    for _, row in tqdm(spots.iterrows()):
        x = int(row[Const.PIX_X]); y = int(row[Const.PIX_Y])
        tile = img[x-half_r:x+half_r, y-half_r:y+half_r]
        masks, flows, styles, diams = model.eval(tile, diameter=None, channels=ch,  cellprob_threshold=cellprob_threshold)
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
            plt.savefig(save_png_result.replace('.', f'_{x}x{y}.'))

    st_adata.obsm["cell_num"] = (ret["cell_num"]).to_numpy()
    st_adata.uns["seg_cell_pos"] = cell_pos
    return st_adata, cell_pos
