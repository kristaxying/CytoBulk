import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
from scipy.sparse import csr_matrix
import cell2location
from docopt import docopt
from cell2location.utils.filtering import filter_genes

def run_cell2location(sc_adata,st_adata,project,ref_run_name,run_name):
    print("data")
    selected = filter_genes(sc_adata, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12)
    sc_adata = sc_adata[:, selected].copy()
    cell2location.models.RegressionModel.setup_anndata(adata=sc_adata,
                        # cell type, covariate used for constructing signatures
                        labels_key='Celltype_minor'
                       )
    mod = cell2location.models.RegressionModel(sc_adata)
    mod.view_anndata_setup()
    mod.train(max_epochs=250, use_gpu=False)
    adata_ref = mod.export_posterior(
            sc_adata, sample_kwargs={'num_samples': 1000, 'batch_size': 2500, 'use_gpu': False}
        )
    mod.save(f"{ref_run_name}", overwrite=False)
    adata_file = f"{ref_run_name}/{project}_sc.h5ad"
    adata_ref.write(adata_file)
    if 'means_per_cluster_mu_fg' in adata_ref.varm.keys():
        inf_aver = adata_ref.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                for i in adata_ref.uns['mod']['factor_names']]].copy()
    else:
        inf_aver = adata_ref.var[[f'means_per_cluster_mu_fg_{i}'
                                for i in adata_ref.uns['mod']['factor_names']]].copy()
    inf_aver.columns = adata_ref.uns['mod']['factor_names']

    intersect = np.intersect1d(st_adata.var_names, inf_aver.index)
    adata_vis = st_adata[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()
    cell2location.models.Cell2location.setup_anndata(adata=adata_vis)
    mod = cell2location.models.Cell2location(
        adata_vis, cell_state_df=inf_aver,
        # the expected average cell abundance: tissue-dependent
        # hyper-prior which can be estimated from paired histology:
        N_cells_per_location=30,
        # hyperparameter controlling normalisation of
        # within-experiment variation in RNA detection:
        detection_alpha=20
    )
    mod.view_anndata_setup()
    mod.train(max_epochs=10000,
        # train using full data (batch_size=None)
        batch_size=None,
        # use all data points in training because
        # we need to estimate cell abundance at all locations
        train_size=1,
        use_gpu=False,
        )

        # plot ELBO loss history during training, removing first 100 epochs from the plot
    adata_vis = mod.export_posterior(
        adata_vis, sample_kwargs={'num_samples': 1000, 'batch_size': mod.adata.n_obs, 'use_gpu': False}
    )
    mod.save(f"{run_name}", overwrite=True)
    adata_file = f"{run_name}/{project}_sp.h5ad"
    adata_vis.write(adata_file)
    adata_vis.obs[adata_vis.uns['mod']['factor_names']] = adata_vis.obsm['q05_cell_abundance_w_sf']
    cell_frac = adata_vis.obs.iloc[:,3:]
    new = cell_frac.values / np.sum(cell_frac.values, axis=1).reshape(-1, 1)
    cell2_frac = pd.DataFrame(new,index=cell_frac.index,columns=cell_frac.columns)
    cell2_frac.to_csv(f"{run_name}/{project}_cell2location_data.csv")

def main():
    project_list = ["NSCLC_GSE179373","KIRC_GSE121636","HNSC_GSE139324"]
    for i in project_list:
      st_data = f"/data1/wangxueying/cytobulk/eval_data/{i}/{i}_expression_test.csv"
      st_meta = f"/data1/wangxueying/cytobulk/eval_data/{i}/meta_test.csv"
      sc_adata = f"/data1/wangxueying/cytobulk/eval_data/{i}/{i}_sc_data_cell2.txt"
      sc_meta = f"/data1/wangxueying/cytobulk/eval_data/{i}/{i}_sc_meta.txt"
      project = i
      results_folder = f"/data1/wangxueying/cytobulk/out/{i}/cell2"
      #st_adata = sc.read_visium(st_adata,library_id="st")
      coordinates = pd.read_csv(st_meta, index_col=0)[["x","y"]]
      st_adata = sc.read_csv(st_data)
      values = np.exp2(st_adata.X)
      values = np.around(values)
      values= values.astype(int)
      st_adata.X = values
      st_adata.obsm["spatial"] = coordinates.to_numpy()
      sc_adata = sc.read_text(sc_adata)
      values = np.exp2(sc_adata.X)
      values = np.around(values)
      values= values.astype(int)
      sc_adata.X = values
      sc_meta = pd.read_csv(sc_meta, index_col=0,sep="\t")
      sc_adata.obs = sc_meta
      sc_adata.obs = sc_adata.obs.rename(columns={'Celltype..minor.lineage.':'Celltype_minor'})
      print(sc_adata.obs.columns)
      if i=="MM_GSE151310":
          sc_adata = sc_adata[sc_adata.obs.Celltype_minor.isin(["B","Th1","Th17","CD8Tcm","MAIT","CD4Tn","CD8Teff","CD8Tex","CD8Tem","cDC1","cDC2","CD8Tn","M1","M2","Mast","Monocyte","NK","pDC","CD4Tconv","Plasma","Tprolif","Treg","Th2"]),:]
      else:
          sc_adata = sc_adata[sc_adata.obs.Celltype_minor.isin(["B","Th1","Th17","CD8Tcm","MAIT","CD4Tn","CD8Teff","CD8Tex","CD8Tem","cDC1","cDC2","CD8Tn","M1","M2","Mast","Monocyte","NK","pDC","CD4Tconv","Plasma","Tprolif","Treg","Tfh","Th2"]),:]
  
      ref_run_name = f'{results_folder}/reference_signatures'
      run_name = f'{results_folder}/cell2location_map'
      run_cell2location(sc_adata,st_adata,project,ref_run_name,run_name)
      


main()

