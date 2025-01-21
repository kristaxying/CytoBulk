import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Const:
    PIX_X = 'X'
    PIX_Y = 'Y'

def rgb2grey(img: np.ndarray):
    return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

def predict_cell_num(st_adata,
                     library_ids="library_ids",
                     project="test",
                     diameter=None,
                     save_png_result=False,
                     model_type='cyto3',
                     out_dir='.',
                     cellprob_threshold=0.4,
                     save=True):
    
    '''
    Predicts cell numbers from spatial transcriptomics data using Cellpose.

    Parameters
    ----------
    st_adata : AnnData
        Spatial transcriptomics AnnData object.

    library_ids : string
        Identifier for the library in the AnnData object.

    diameter : int, optional
        Estimated diameter of the cells to be detected.

    save_png_result : bool, optional
        If True, saves the segmentation results as PNG files.

    model_type : string, optional
        Type of Cellpose model to use. Default is 'cyto3'.

    out_path : string, optional
        Directory to save the output files. Default is './'.

    cellprob_threshold : float, optional
        Cell probability threshold for Cellpose. Default is 1.

    save : bool, optional
        If True, saves the results to an h5ad file. Default is True.

    Returns
    -------
    st_adata : AnnData
        Updated AnnData object with predicted cell numbers.

    cell_pos : pandas.DataFrame
        DataFrame containing cell positions.
    '''
    from cellpose import models, io, plot
    from tqdm import tqdm

    print('-----Initializing model...')
    model = models.Cellpose(model_type=model_type)
    ch = [0, 0] # NOTE: here we set all images to greyscale

    print('-----Reading files...')
    img = rgb2grey(st_adata.uns['spatial'][library_ids]['images']['hires'])


    coord = st_adata.obsm['spatial']*st_adata.uns['spatial'][library_ids]['scalefactors']['tissue_hires_scalef']
    spots = pd.DataFrame(coord, columns=["X", "Y"])
    crop_r = int(st_adata.uns['spatial'][library_ids]['scalefactors']['spot_diameter_fullres']*st_adata.uns['spatial'][library_ids]['scalefactors']['tissue_hires_scalef'])
    #crop_r = int(st_adata.uns['spatial'][library_ids]['scalefactors']['spot_diameter_fullres'])
    half_r = crop_r // 2 + 5
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
            plt.savefig(f"{out_dir}/figures/{_}_segmentation_result.png")

    st_adata.obsm["cell_num"] = (ret["cell_num"]).to_numpy()
    st_adata.uns["seg_cell_pos"] = cell_pos
    ret["cell_num"].to_csv(f"{out_dir}/output/{project}_cell_num.csv")
    cell_pos.to_csv(f"{out_dir}/output/{project}_cell_pos.csv")
    st_adata.write_h5ad(f"{out_dir}/output/segmentation_adata.h5ad")
    return st_adata, cell_pos
