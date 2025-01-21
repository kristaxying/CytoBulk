"""
Accessors for frequently used data within an :class:`~anndata.AnnData`.
"""

# expose the API
from ._plot import plot_batch_effect as batch_effect
from ._plot import plot_celltype_fraction_pie as celltype_fraction_pie
from ._plot import plot_celltype_fraction_heatmap as celltype_fraction_heatmap
from ._plot import plot_paired_violin as paired_violin
from ._plot import plot_reconstruction as reconstruction_corr
from ._plot import plot_he_cell_type as he_cell_type
from ._plot import plot_gene_similarity as gene_similarity