---
author: "momo"
date: 2022-05-11
title: "自监督med论文笔记"
categories: [
    "论文笔记",
    "医学图像分析",
    "ICML",
]
---

# Momentum contrastive learning for few-shot COVID-19 diagnosis from chest CT images

https://www.sciencedirect.com/science/article/pii/S0031320321000133?via%3Dihub

2021 pattern recognition

作者开发了一个端到端的可训练的深度few-shot学习框架，可以在胸部CT图像上以最少的训练提供准确的预测。具体来说，首先使用实例判别任务来强制模型判别两幅图像是否为同一实例。然后生成相同图像的不同视图以增加原始数据集。由于在这一阶段的目标是增加除辨别力之外的变异，能够有效地避免前面提到的数据增强的缺点（过拟合、对参数高度敏感）。然后，部署一个自监督的策略[22]，用动力对比训练来进一步提高性能。建立一个动态字典来进行（key,query）查询，其中的key是从数据中采样并由编码器编码的。然而，由于反向传播的原因，字典中的键是有噪声的，而且不一致[23]。采用动量机制，通过在不同尺度上更新key和query encoders来缓解这种影响。最后，我们利用两个公共的肺部数据集来预训练一个嵌入网络，并采用prototypical network[24]来进行few-shot分类，通过测量与每个类别的衍生原型表示的距离来学习分类的度量空间。在两个新的数据集上进行的广泛实验表明，我们的模型为用非常有限的可用训练数据进行COVID-19的快速诊断提供了一个有希望的工具。

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




