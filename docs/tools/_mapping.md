# tools._mapping

----------
```python
def bulk_mapping(frac_data,
                bulk_adata,
                sc_adata,
                n_cell=100,
                annotation_key="curated_cell_type",
                bulk_layer=None,
                sc_layer=None,
                reorder=True,
                multiprocessing=True,
                cpu_num=cpu_count()-2,
                dataset_name="",
                out_dir=".",
                normalization=True,
                filter_gene=True,
                cut_off_value=0.6,
                save=True)
```
::: cytobulk.tools._mapping.bulk_mapping
