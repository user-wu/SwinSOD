# SwinSOD
Code release for paper "SwinSOD: Salient Object Detection using Swin-Transformer"

## Introduction

## Requirements
* python 3.6
* PyTorch >= 1.7
* torchvision >= 0.4.2
* PIL
* Numpy

## Data Preparation
Download the following datasets and unzip them into data folder
* [PASCAL-S](https://ccvl.jhu.edu/datasets/)
* [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)
* [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency)
* [DUT-OMRON](http://saliencydetection.net/dut-omron/)
* [DUTS](http://saliencydetection.net/duts/)
## Directory Structure
```
 data --------------------------
      |-DUTS        -image/
      |             -mask/
      |             -test.txt
      |             -train.txt
      --------------------------
      |-DUT-OMRON   -image/
      |             -mask/
      |             -test.txt
      --------------------------
      |-ECSSD       -image/
      |             -mask/
      |             -test.txt
      --------------------------
      |-HKU-IS      -image/
      |             -mask/
      |             -test.txt
      --------------------------
      |-PASCAL-S    -image/
      |             -mask/
      |             -test.txt
      --------------------------
```
## Train
* Clone repository
```
git clone https://github.com/user-wu/SwinSOD.git
cd SwinSOD/src/
```
* Train
```
python train.py
```
## Inference
```
cd SwinSOD/src/
python test.py
```

## Citation

The code will upload soon.
