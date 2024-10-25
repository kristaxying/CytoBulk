# plots._plot

----------
```python
plot_celltype_fraction_pie(adata,
                               scale_facter_x=1,
                               scale_factor_y=1,
                               r=0.1,
                               out_dir='/plot',
                               project_name='project',
                               color_list=None,
                               color_map="Spectral",
                               rotation_angle=None,
                               figsize=(2, 2))
```
::: cytobulk.plots._plot.plot_celltype_fraction_pie

----------
```python
plot_celltype_fraction_heatmap(adata,
                                   label,
                                    r=0.1,
                                    out_dir='/plot',
                                    project_name='project',
                                    color_map='crest',
                                    rotation_angle=None,
                                    figsize=(2.7, 2))
```
::: cytobulk.plots._plot.plot_celltype_fraction_heatmap

----------
```python

plot_paired_violin(adata,
                       label,
                       gene,
                       stats_method='spearmanr',
                       out_dir='/plot',
                       color_list=Const.DIFF_COLOR_N,
                       project_name="test",
                       figsize=(6, 4),
                       ylim=[-0.1, 1.2])
```
::: cytobulk.plots._plot.plot_paired_violin

----------
```python

def plot_reconstruction(adata,
                        out_dir,
                        project_name="test",
                        rotation_angle=None,
                        spot_size=0.5):
```
::: cytobulk.plots._plot.plot_reconstruction
