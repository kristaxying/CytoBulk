# Overview
CytoBulk aims to Integrate transcriptional and image data to depict the tumor microenvironment accurately with the graph frequency domain model


# Documentation

To install and use CytoBulk, please visit https://kristaxying.github.io/CytoBulk/

# System requirements

We have tested the package on the following systems:
- Linux: Ubuntu 20 (GPU 3080)
- Windows: Windows 11 Enterprise (CPU)

# Installation Guide

Follow the steps below to install and set up **CytoBulk**.

---

## Setting Up the Environment for Python and R
The CytoBulk package is developed based on the pytorch framework and can be implemented on both GPU and CPU. We recommend running the package on GPU. Please ensure that pytorch and CUDNN are installed correctly.
### Option 1: Set Python and R Together
```
conda config --append channels conda-forge
conda create --name cytobulk python=3.10 r-base=4.4
conda activate cytobulk
pip install cytobulk
```
This approach is suitable for users who want all dependencies managed within the same Conda environment. However, it might **not work reliably on Windows** due to potential issues with R configuration in Conda.
**If you have installed cytobulk package, please run following code to update the latest version**
```
pip install cytobulk==0.1.20

```
### Option 2: Set Only Python and Specify R Path Separately

```
conda create --name cytobulk python=3.10
conda activate cytobulk
pip install cytobulk

```
Then, before running the main program, you need to specify the path to your locally installed R. This can be done using Python by setting the R_HOME environment variable. Add the following lines at the beginning of your Python script:
```
import os
# Set the R installation path (adjust the path based on your R installation)
os.environ['R_HOME'] = r_path
```
**If you have installed cytobulk package, please run following code to update the latest version**
```
pip install cytobulk==0.1.20

```
## Install required R packages

 To run **CytoBulk**, make sure all the following **prerequisites** are installed.


**R 4.4.0 or higher and the following packages** 

- Giotto (1.1.2) <https://giottosuite.readthedocs.io/en/master/gettingstarted.html>
- scran (1.32.0) <https://bioconductor.org/packages/release/bioc/html/scran.html>
- sva (3.52.0) <https://www.bioconductor.org/packages/release/bioc/html/sva.html>


# Run demo
Please visit Examples section at https://kristaxying.github.io/CytoBulk/.



### Maintainer
WANG Xueying xywang85-c@my.cityu.edu.hk


### The Latest Version
0.1.20  April 1, 2025