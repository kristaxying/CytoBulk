# CytoBulk
The algorithm for mapping bulk data to spatial HE image.


## Usage

For testing the CytoBulk class, you may refer the test.py file.
- CytoBulk is the major class which contains the full framework.
- graph_deconv offer the function of deconvolution and mapping sc to bulk file.
- image_prediction could predict the cell type and expression from HE image.
- spatial_mapping use spot data and mapped sc data to reconstruct the spot data at single cell resolution. Then will refine the single cell coordinates based on HE cell segmentation results.
- utils give the basic functions.

## 




### Maintainer
WANG Xueying xywang85-c@my.cityu.edu.hk

WANG Yian 18019349217@163.com