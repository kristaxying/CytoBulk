# Installation Guide

Follow the steps below to install and set up **CytoBulk**.

---

## Setting Up the Environment for Python and R

### Option 1: Install Python and R Together
```
conda create --name cytobulk python=3.10 r-base=4.4
```
This approach is suitable for users who want all dependencies managed within the same Conda environment. However, it might **not work reliably on Windows** due to potential issues with R configuration in Conda.
### Option 2: Install Only Python and Specify R Path Separately

```
conda create --name cytobulk python=3.10

```
Then, before running the main program, you need to specify the path to your locally installed R. This can be done using Python by setting the R_HOME environment variable. Add the following lines at the beginning of your Python script:
```
import os
# Set the R installation path (adjust the path based on your R installation)
os.environ['R_HOME'] = r_path
```
## Prerequisites

The CytoBulk package is developed based on the pytorch framework and can be implemented on both GPU and CPU. We recommend running the package on GPU. Please ensure that pytorch and CUDNN are installed correctly. To run **CytoBulk**, make sure all the following **prerequisites** are installed.


1. **R 4.4.0 or higher and the following packages**
     - Giotto (1.1.2) <https://giottosuite.readthedocs.io/en/master/gettingstarted.html>
     - scran (1.32.0) <https://bioconductor.org/packages/release/bioc/html/scran.html>
     - sva (3.52.0) <https://www.bioconductor.org/packages/release/bioc/html/sva.html>


2. **python 3.10 or higher**
     - pytorch (2.5.1) <https://pytorch.org/get-started/locally/>
     - cuda (11.8 or later) -- if using GPU

## Installation

```
conda activate cytobulk
```

```
pip install cytobulk==0.1.5

or

git clone git@github.com:kristaxying/CytoBulk.git

cd CytoBulk

python setup.py build

python setup.py install --user`

```