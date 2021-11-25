# Multi-Scale-Unsupervised-Framework-MSUF
This page is undergoing editing, the data will be uploaded later.
## Preparation
Our code is performed in Pytorch 1.8.0 basis on Python 3.8.
## Introduction
network.py: Our DNN architectures, implemented on three scales.

generation.py:  Generate the trainging or testing data (image pairs) by your own datasets.

loss.py: Store various loss functions.

train.py : Training Process.

test.py: Testing Process.

STN.py: Affine Transformation based on STN.

CFOG.cp38-win_amd64.pyd: Similarity evaluation for multi-modal images is recommand as the NCC of the Chanel Feature of Orientated Gradient (CFOG). We provide this executable version of the CFOG descriptor, You could use 'import CFOG' to call this descriptor or adopt thr other descriptors (i.e. HOPC, HOG or LSS) in your code.
