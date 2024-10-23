"""
General utility functions
"""

# expose the API


from ._math import get_sum, pca,normalization_cpm,pear,rotate_matrix,convert_pvalue_to_asterisks
from ._math import normal_center_df as normal_center
from ._stimulation import bulk_simulation,bulk_simulation_case,st_simulation_case,st_simulation
from ._utils import compute_cluster_averages,compute_bulk_with_average_exp, data_dict_integration,filter_samples,compute_average_cosin,marker_integration,filter_gene,normalize_data
from ._read_data import check_paths
from ._evaluation import eval_fraction,eval_comparsion,eval_each_sample_mse