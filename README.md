# Certified Robustness for Deep Equilibrium Models via Serialized Random Smoothing
The official code of the NeurIPS 2024 paper: [Certified Robustness for Deep Equilibrium Models via Serialized Random Smoothing](https://arxiv.org/abs/2411.00899) by [Weizhi Gao](), [Zhichao Hou](https://weizhigao.github.io/), [Han Xu](https://sites.google.com/view/han-xu-123/home), and [Xiaorui Liu](https://sites.google.com/ncsu.edu/xiaorui/). This repo is built upon [Randomized Smoothing](https://github.com/locuslab/smoothing) and [Deep Equilibrium Models](https://github.com/locuslab/deq).

## Introduction

Serialized Randomized Smoothing is the first work studying randomized smoothing for Deep Equilibrium Models and accelerte it. We make use of randomized smoothing to certify DEQ, showing SOTA certified robustness. Our serialized randomized smoothing can accelerate randomized smoothing on DEQ up to 7x almost without sacrificing the certified accuracy. It is also the first work that extend the certification robustness of DEQ to ImageNet. 

The results on CIFAR-10.
| Model \ Radius | 0.0  | 0.25 | 0.5  | 0.75 | 1.0  | 1.25 | 1.5  | Time (s)    |
|----------------|------|------|------|------|------|------|------|-------------|
| MDEQ-1A        | 28%  | 19%  | 13%  | 8%   | 5%   | 3%   | 1%   | 1.06        |
| MDEQ-5A        | 50%  | 41%  | 32%  | 21%  | 15%  | 10%  | 6%   | 2.59        |
| MDEQ-30A       | **67%** | **55%** | 45%  | 33%  | 23%  | 16%  | 12%  | 12.89       |
| SRS-MDEQ-1N    | 61%  | 52%  | 44%  | 31%  | 22%  | 15%  | 11%  | 1.02 (**13×**) |
| SRS-MDEQ-1A    | 63%  | 53%  | **45%** | 32%  | 22%  | 16%  | **12%** | 1.79 (**7×**) |
| SRS-MDEQ-3A    | **66%** | 54%  | **45%** | **33%** | 23%  | 16%  | 11%  | 2.55 (**5×**) |

The results on ImageNet.
| Model \ Radius  | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | Time (s)       |
|-----------------|-----|-----|-----|-----|-----|-----|-----|----------------|
| MDEQ-1B         | 2%  | 2%  | 1%  | 1%  | 1%  | 1%  | 0%  | 7.30           |
| MDEQ-5B         | 39% | 33% | 28% | 23% | 19% | 15% | 11% | 31.77          |
| MDEQ-14B        | **45%** | 39% | 33% | 28% | 22% | 17% | 11% | 88.33       |
|                 |     |     |     |     |     |     |     |                |
| SRS-MDEQ-1B     | 40% | 34% | 32% | 27% | 21% | 16% | 10% | 15.21 (**6×**) |
| SRS-MDEQ-3B     | 44% | **39%** | **33%** | **28%** | **22%** | **17%** | **11%** | 27.48 (**3×**) |

## Getting Started

Our code is based on Pytorch. You can build the environment with the following steps.

```
git clone git@github.com:WeizhiGao/Serialized-Randomized-Smoothing.git
conda create -n srs
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install scipy pandas statsmodels matplotlib seaborn
conda install conda-forge::tensorboard termcolor yacs
cd Serialized-Randomized-Smoothing
```

## Running the code
* First train the Gaussian augmented deep equilibrium models. For instance, you can train MDEQ-LARGE under the noise level 0.50 with the following code:
```
cd DEQ/MDEQ-Vision
python ./tools/cls_train.py \
 --logDir ./log/cifar10/mdeq/noise_0.50 \
 --outDir ./output/cifar10/noise_0.50 \
 --noise 0.50 \
 --cfg ./experiments/cifar/cls_mdeq_LARGE_reg.yaml
```

* Constuct the model directory for SRS as follows:

```
DEQ
|--- code
+--- model
       +--- cifar10
       |       | cls_mdeq_LARGE
       |       | cls_mdeq_SMALL
       +--- imagenet
       +       + cls_mdeq_SMALL_imagenet
```
Then copy the corresponding models to the directory named by noise levels. For instance, the model trained with noise level 0.50 will be called as ```noise50.pth.tar```

* Run the certification code with the following commands:
```
cd ../../SRS
python code/certify.py cifar10 models/cifar10/cls_mdeq_LARGE/noise50.pth.tar 0.50 data/certify/cifar10/SRS-MDEQ-3A.txt --skip 20 --batch 400 --N 10000 \
     --srs --warmup_thresh 10 --warmup_solver anderson \
     --f_solver anderson --f_thresh 3 --conf_drop 400
```
The example is to certify MDEQ (trained with noise level 0.5) on CIFAR10 with the default hyperparmeters reported in our paper.

## Citation
If you find our work helpful, feel free to give us a cite.
```
@article{gao2024certified,
  title={Certified Robustness for Deep Equilibrium Models via Serialized Random Smoothing},
  author={Gao, Weizhi and Hou, Zhichao and Xu, Han and Liu, Xiaorui},
  journal={arXiv preprint arXiv:2411.00899},
  year={2024}
}
```
