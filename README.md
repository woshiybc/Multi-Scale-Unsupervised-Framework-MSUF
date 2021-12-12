# Multi-Scale-Unsupervised-Framework-MSUF
This is an implementation of our paper: A Multi-Scale Framework with Unsupervised Learning for Remote Sensing Image Registration.
![Proposed Framework in the Paper](https://github.com/yeyuanxin110/Multi-Scale-Unsupervised-Framework-MSUF/blob/main/MSUF.png)
## Preparation
Our code is performed in Pytorch 1.8.0 basis on Python 3.8.
## Introduction
network.py: Our DNN architectures, implemented on three scales.

generation.py:  Generate the trainging or testing data (image pairs) by datasets provided by the paper or your own datasets.

loss.py: Store various loss functions.

train.py : Training Process.

test.py: Testing Process.

STN.py: Affine Transformation based on STN.

CFOG.cp38-win_amd64.pyd: Similarity evaluation for multi-modal images is recommand as the NCC of the Chanel Feature of Orientated Gradient (CFOG). We provide this executable version of the CFOG descriptor, You could use 'import CFOG' to call this descriptor or adopt thr other descriptors (i.e. HOPC, HOG or LSS) in your code.
## Datasets
The multi-modal original image pairs adopted in the paper have been uploaded to Google Drive. You could download them and put them into generation.py to generate the training or testing image pairs.

![Optical-Optical dataset](https://github.com/yeyuanxin110/Multi-Scale-Unsupervised-Framework-MSUF/blob/main/Optical-Optical.png)
Optical-Optical dataset: https://drive.google.com/file/d/1U0fpCnizcl33TgdRwvfQpqOr1Ojcj6a9/view?usp=sharing

![Optical-Infrared dataset](https://github.com/yeyuanxin110/Multi-Scale-Unsupervised-Framework-MSUF/blob/main/Optical-Infrared.png)
Optical-Infrared dataset: https://drive.google.com/file/d/1c4Ao4CoMerntNVf2Qn3hY0eEtwURh8iM/view?usp=sharing

![Optical-SAR dataset](https://github.com/yeyuanxin110/Multi-Scale-Unsupervised-Framework-MSUF/blob/main/Optical-SAR.png)
Optical-SAR dataset: https://drive.google.com/file/d/181IEtG6ciBsQGhM6TgEDfv8yglAWsKxy/view?usp=sharing

![Optical-RasterMap dataset](https://github.com/yeyuanxin110/Multi-Scale-Unsupervised-Framework-MSUF/blob/main/Optical-Map.png)
Optical-RasterMap dataset: https://drive.google.com/file/d/1kIqXy3-KCTLwaPaxTrEFKSt49LvZnWAU/view?usp=sharing
