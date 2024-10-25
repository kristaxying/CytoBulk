# tools._deconv

----------
```python
def bulk_deconv(bulk_data,
                sc_adata,
                annotation_key,
                marker_data=None,
                rename=None,
                dataset_name="",
                out_dir='.',
                different_source=True,
                cell_list=None,
                scale_factors=10000,
                trans_method="log",
                save = True,
                save_figure=True,
                n_cell=100,
                **kwargs)
```
::: cytobulk.tools._deconv.bulk_deconv

----------
```python
st_deconv(st_adata,
            sc_adata,
            annotation_key,
            marker_list=None,
            rename=None,
            dataset_name="",
            out_dir='.',
            different_source=True,
            cell_list=None,
            scale_factors=10000,
            trans_method="log",
            save = True,
            save_figure=True,
            n_cell=10,
            **kwargs)
```
::: cytobulk.tools._deconv.st_deconv