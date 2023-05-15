import pandas as pd
import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix
from utils import *
from cell2location.utils.filtering import filter_genes
from cell2location.models import RegressionModel
import cell2location

class SpatialMapping:
    def __init__(self,st_data,st_meta,sc_data,sc_meta):
        """
            st_data: dataframe, the spatial gene expression data.
        """
        self.st_data = st_data
        self.st_meta = st_meta
        self.sc_data = sc_data
        self.sc_meta = sc_meta
    
    def preprocessing(self):
        """
            Reoriginze the spatial data with annodata format and execute basic preprocessing.
        """
        normback_sc = norm_back(self.sc_data)
        self.st_adata = reorignize_data(self.st_data,self.st_meta)
        sc_adata = reorignize_data(self.sc_data.T,self.sc_meta,normback_sc)
        selected = filter_genes(sc_adata, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12)
        self.sc_adata = sc_adata[:, selected].copy()
        print(self.sc_adata)
        self._cell_type_deconvolution()

        



    def _cell_type_deconvolution(self):
        """
            input is the 
        """
        '''
        cell2location.models.RegressionModel.setup_anndata(adata=self.sc_adata,
                        # cell type, covariate used for constructing signatures
                        labels_key='cell_type',
                       )
        
        mod = RegressionModel(self.sc_adata)
        mod.view_anndata_setup()
        mod.train(max_epochs=250, use_gpu=True)
        mod.plot_history(20)
        adata_ref = mod.export_posterior(
            self.sc_adata, sample_kwargs={'num_samples': 1000, 'batch_size': 2500, 'use_gpu': True}
        )
        print(adata_ref)
        adata_file = f"CytoBulk/CytoBulk/out/sc.h5ad"
        if 'means_per_cluster_mu_fg' in adata_ref.varm.keys():
            inf_aver = adata_ref.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                    for i in adata_ref.uns['mod']['factor_names']]].copy()
        else:
            inf_aver = adata_ref.var[[f'means_per_cluster_mu_fg_{i}'
                                    for i in adata_ref.uns['mod']['factor_names']]].copy()
        inf_aver.columns = adata_ref.uns['mod']['factor_names']

        intersect = np.intersect1d(self.st_adata.var_names, inf_aver.index)

        adata_vis = self.st_adata[:, intersect].copy()
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
        mod.train(max_epochs=30000,
            # train using full data (batch_size=None)
            batch_size=None,
            # use all data points in training because
            # we need to estimate cell abundance at all locations
            train_size=1,
            use_gpu=True,
            )

        # plot ELBO loss history during training, removing first 100 epochs from the plot
        adata_vis = mod.export_posterior(
            adata_vis, sample_kwargs={'num_samples': 1000, 'batch_size': mod.adata.n_obs, 'use_gpu': True}
        )

        # Save model
        mod.save(f"CytoBulk/CytoBulk/out", overwrite=True)

        # mod = cell2location.models.Cell2location.load(f"{run_name}", adata_vis)

        # Save anndata object with results
        '''
        
        mean_sc = cell2location.cluster_averages.compute_cluster_averages(adata=self.sc_adata, labels='cell_type')
        intersect = np.intersect1d(self.st_adata.var_names, mean_sc.index)
        adata_vis = self.st_adata[:, intersect].copy()
        print(adata_vis)
        mean_sc = mean_sc.loc[intersect, :].copy()
        cell2location.models.Cell2location.setup_anndata(adata=adata_vis, batch_key="sample")
        mod = cell2location.models.Cell2location(
            adata_vis, cell_state_df=mean_sc,
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
          use_gpu=True,
         )
        adata_vis = mod.export_posterior(
            adata_vis, sample_kwargs={'num_samples': 1000, 'batch_size': mod.adata.n_obs, 'use_gpu': True}
        )

        # Save anndata object with results
        mod.save(f"CytoBulk/CytoBulk/out", overwrite=True)
        

        
        adata_file = "CytoBulk/CytoBulk/out/sp.h5ad"
        adata_vis.write(adata_file)
        mod.plot_QC()

        
        
    def get_expression():
        """
            
        """