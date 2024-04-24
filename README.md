# SwinSOD
Code release for paper "SwinSOD: Salient Object Detection using Swin-Transformer" 

by Shuang Wu, Guangjian Zhang, Xuefeng Liu

* !!News: The [SSRN Preprinted paper](https://ssrn.com/abstract=4556674)  "SwinSOD: Salient Object Detection using Swin-Transformer" has been accepted by [《Image and Vision Computing》](https://www.sciencedirect.com/science/article/abs/pii/S0262885624001434?via%3Dihub).




## Introduction
The Transformer structure has achieved excellent performance in a wide range of applications in the field of computer vision, and Swin Transformer also shows strong feature representation capabilities. On this basis, we proposed a fusion model SwinSOD for RGB salient object detection. This model used Swin-Transformer as the encoder to extract hierarchical features, was driven by a multi-head attention mechanism to bridge the gap between hierarchical features, progressively fused adjacent layer feature information under the guidance of global information, and refined the boundaries of saliency objects through the feedback information. Specifically, the Swin-Transformer encoder extracted multi-level features and then recalibrates the channels to optimize intra-layer channel features. The feature fusion module realized feature fusion between each layer under the guidance of global information. In order to clarify the fuzzy boundaries, the second stage feature fusion achieved edge refinement under the guidance of feedback information. The proposed model outperforms state-of-the-art models on five popular SOD datasets, demonstrating the advanced performance of this network.<img width="1111" alt="image" src="https://github.com/user-wu/SwinSOD/assets/67259115/1844511e-4570-4982-84d6-ae5d77bbb17d">

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
* [Pretrained model](https://pan.baidu.com/s/13wu5qtljqFOUvdYSOV1STg?pwd=z5l0).
## Inference
```
cd SwinSOD/src/
python test.py
```
* After testing, saliency maps of PASCAL-S, ECSSD, HKU-IS, DUT-OMRON, DUTS-TE will be saved in ```eval/maps/``` folder.
* Trained model: [model](https://pan.baidu.com/s/1-l8iNwEOOq9Y5CO9TYghKw?pwd=e4na)
* Saliency maps for reference: [saliency maps](https://pan.baidu.com/s/1URFCxG7JIVP6u_mHoaTo6g?pwd=3zfq)

## Citation
* If you find this work is helpful, please cite our paper

later release……
```
Wu, Shuang and Zhang, Guangjian and Liu, Xuefeng, Swinsod: Salient Object Detection Using Swin Transformer.
Available at SSRN: https://ssrn.com/abstract=4556674 or http://dx.doi.org/10.2139/ssrn.4556674
```
