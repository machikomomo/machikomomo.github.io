---
author: "momo"
date: 2022-05-11
title: "CL_PLP论文笔记"
categories: [
    "论文笔记",
    "医学图像分析",
]
---

# Deep semi-supervised learning with contrastive learning and partial label propagation for image data

https://www.sciencedirect.com/science/article/pii/S0950705122002702

在本文中，我们提出了一种新的基于对比性自监督学习和部分标签传播策略的深度半监督学习算法，称为CL_PLP。该方法由两个模块组成，包括自监督特征提取模块和部分标签传播模块，可以分别改进传统标签传播方法的两个阶段。当标签不足时，经过训练的网络很难学到准确的特征表示，这就进一步降低了标签传播阶段产生的伪标签的准确性。为了解决这个问题，我们提出了一个新的网络结构，增加了投影层和一个额外的对比性损失项，用于对比性学习。同时，我们通过结合强增强和弱增强[6]来扩展数据集，以增加数据集的多样性和模型的稳健性。在第二阶段，我们通过根据伪标签的质量中断标签传播程序来提高高置信度的伪标签的影响。最后，我们提出了一个策略，将我们的部分标签传播模块与最先进的归纳式半监督学习算法相结合。

我们的算法在三个标准基线数据集CIFAR-10、CIFAR-100和miniImageNet上的表现优于之前最先进的传导式深度半监督学习方法。通过将我们的模型转移到医学COVID19-Xray数据集，它也取得了良好的性能。最后，我们提出了一种策略，将我们的部分标签传播模块与归纳式半监督学习方法结合起来。

## experiment

We evaluated our proposed model using two publicly available annotated COVID-19 CT slices datasets: (1) COVID-19 CT and (2) a dataset provided by the Italian Society of Medical and Interventional Radiology4 and preprocessed by MedSeg.5 It is worth mentioning that there is no overlap between COVID-19 CT and MegSeg as they come from different countries. When dividing the support and query sets for classification, we divided the datasets at a patient-level instead of CT level to avoid any possible overlap. （2个公开数据集，covid19 CT 和 MedSeg。来自两个不同的国家，没有重叠。在病人层面而不是ct层面划分数据集）

https://github.com/UCSD-AI4H/COVID-CT

http://medicalsegmentation.com/covid19/

![截屏2022-05-12 下午1.41.04](/Users/momochan/Library/Application Support/typora-user-images/截屏2022-05-12 下午1.41.04.png)

两个数据集合并。所有slide通过opencv处理成512*512。

适当的预训练。与其他现有的方法，如Self-Trans[20]，使用ImageNet来预训练模型不同，作者采用了DeepLesion[40]和肺部图像数据库联盟图像集（LIDC-IDRI）7。DeepLesion包含超过32000张肺部CT图像，而LIDC-IDRI包括244,617张。这两个数据集都是公开的，并以肺部疾病为重点。使用这两个没有标签的数据集来预训练编码器网络。

参数：

we used the SGD optimizer with a weight decay of 0.0001 and momentum of 0.9

The momentum update coefficient was 0.999. 动量更新系数。

mini-batch size was set to 256 in eight GPUs

The number of epochs was 200. The initial learning rate was 0.03, which was then multiplied by 0.1 after 120 and 160 epochs, as described in [35]. 

ResNet-50 was used as the encoder.

The two- layer MLP projection head included a 2048-D hidden layer with a ReLU activation function.

The weights were initialized by using He initialization [41], and the temperature parameter τ was set to 0.07. 

The experiments were conducted on eight GPUS which includes six NVIDIA TITAN X Pascal GPUs and two NVIDIA TITAN RTX.



