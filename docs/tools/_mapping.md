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



----------
```python
def st_mapping(st_adata,
               sc_adata,
               out_dir,
               project,
               annotation_key,
               **kwargs)
```
::: cytobulk.tools._mapping.st_mapping



----------
```python
def he_mapping(image_dir,
               out_dir,
               project,
               lr_data = None,
               sc_adata = None,
               annotation_key="curated_celltype",
               k_neighbor=30,
               alpha=0.5,
               mapping_sc=True,
               **kwargs)
```
::: cytobulk.tools._mapping.he_mapping