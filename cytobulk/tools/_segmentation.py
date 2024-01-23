#from cellpose.cellpose import models, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from cellpose.cellpose import plot
from tqdm import tqdm


class Const:
    PIX_X = 'X'
    PIX_Y = 'Y'

def rgb2grey(img: np.ndarray):
    return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

# TODO: deal with the vesiculars
def predict_cell_num(img_path, 
                     spot_path,
                     crop_r,
                     save_png_result=None,
                     model_type='cyto2', 
                     cellprob_threshold=1):
    '''
    Params:
    -----
    img_path: str, WSI. size can be arbitrary.
    spot_path: str. csv/tsv file contains the spot pixel coordinates.
    crop_r: the length of the tile. each tile will be a square.
    save_png_result: None or str, if None, do not save the segmentation result to a png file; otherwise, save the png to the provided file path name.

    Returns:
    -----
    a pd.Dataframe with 'X', 'Y', 'cell_num' columns.
    '''

    print('-----Initializing model...')
    model = models.Cellpose(model_type=model_type)
    ch = [0, 0] # NOTE: here we set all images to greyscale

    print('-----Reading files...')
    img = rgb2grey(io.imread(img_path))
    spots = pd.read_csv(spot_path, sep=',' if '.csv' in spot_path else '\t').iloc
    half_r = crop_r // 2
    
    print('-----Predicting cell number...')
    ret = pd.DataFrame(data={'X':[], 'Y':[], 'cell_num':[]})
    for _, row in tqdm(spots.iterrows()):
        x = int(row[Const.PIX_X]); y = int(row[Const.PIX_Y])
        tile = img[x-half_r:x+half_r, y-half_r:y+half_r]
        masks, flows, styles, diams = model.eval(tile, diameter=None, channels=ch,  cellprob_threshold=cellprob_threshold)
        cell_num = len(np.unique(masks))
        ret.loc[len(ret.index)] = [x, y, cell_num]

        if save_png_result:
            fig = plt.figure()
            plot.show_segmentation(fig, tile, masks, flows[0], channels=ch)
            plt.tight_layout()
            plt.savefig(save_png_result.replace('.', f'_{x}x{y}.'))

    return ret


