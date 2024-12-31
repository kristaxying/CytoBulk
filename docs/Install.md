# Installation Guide

Follow the steps below to install and set up **CytoBulk**.

---

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

pip install CytoBulk

or

git clone git@github.com:kristaxying/CytoBulk.git

cd CytoBulk

python setup.py build

python setup.py install --user