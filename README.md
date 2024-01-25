# CytoBulk
The algorithm for mapping bulk data to spatial HE image.


## Usage

For testing the CytoBulk class, you may refer the test.py file.
- CytoBulk is the major class which contains the full framework.
- graph_deconv offer the function of deconvolution and mapping sc to bulk file.
- image_prediction could predict the cell type and expression from HE image.
- spatial_mapping use spot data and mapped sc data to reconstruct the spot data at single cell resolution. Then will refine the single cell coordinates based on HE cell segmentation results.
- utils give the basic functions.

- please setup git modules before using this repo:
```bash
git submodule update --init --recursive
```

- to check the doc, run
```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

## 




### Maintainer
WANG Xueying xywang85-c@my.cityu.edu.hk

WANG Yian yianwang5-c@my.cityu.edu.hk
