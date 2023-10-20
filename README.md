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
* Swin-Transformer is used as the backbone of SwinSOD and DUTS-TR is used to train the model
* batch=32, lr=0.05, momen=0.9, decay=5e-4, epoch=32
* Warm-up and linear decay strategies are used to change the learning rate lr
* After training, the result models will be saved in out folder
## Inference
```
cd SwinSOD/src/
python test.py
```
* After testing, saliency maps of PASCAL-S, ECSSD, HKU-IS, DUT-OMRON, DUTS-TE will be saved in eval/maps/ folder.
* Trained model: [model](Soon realease……)
* Saliency maps for reference: [saliency maps](Soon realease……)

## Citation
* If you find this work is helpful, please cite our paper
```
@article{wu4556674swinsod,
  title={Swinsod: Salient Object Detection Using Swin Transformer},
  author={Wu, Shuang and Zhang, Guangjian and Liu, Xuefeng},
  journal={Available at SSRN 4556674}
}
```
The code will upload soon.
