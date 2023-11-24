# utils._stimulation

----------
```python
def bulk_simulation(sc_adata,
                    cell_list,
                    annotation_key,
                    project,
                    out_dir,
                    n_sample_each_group=100,
                    min_cells_each_group=100,
                    cell_gap_each_group=100,
                    group_number=5,
                    rename_dict=None,
                    save=False,
                    return_adata=True)
```
::: cytobulk.utils.bulk_simulation

----------
```python
def bulk_simulation_case(sc_adata,
                        cell_list,
                        annotation_key,
                        project,
                        out_dir,
                        n_sample_each_group=100,
                        min_cells_each_group=100,
                        cell_gap_each_group=100,
                        group_number=5,
                        rename_dict=None,
                        save=True,
                        scale_factors=100000,
                        trans_method="log",
                        return_adata=False)
```
::: cytobulk.utils.bulk_simulation_case

