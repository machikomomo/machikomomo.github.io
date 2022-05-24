---
author: "momo"
date: 2022-05-24
title: "MAG-SD论文笔记"
categories: [
    "论文笔记",
    "JBHI",
]
---

# MAG-SD

https://github.com/lijx1996/MAG-SD

https://ieeexplore.ieee.org/abstract/document/9351607

2021 JBHI

## 我的启发

之前对WS-DAN和MAG-SD做过一个比较。这次先回顾一下之前记录的比较，然后再重新读一下这篇文章并写笔记（主要是看一下它的框架是怎么修改的）。最后看代码。

## 比较

- 1、特征图的多尺度。使用ResNet50作为backbone提取出不同尺寸的feature map——f1，f2，f3。生成对应的a1、a2、a3。（For ResNet50 we used, feature maps with 512 ∗ 28 ∗ 28, 1024 ∗ 14 ∗ 14, 2048 ∗ 7 ∗ 7 sizes are chosen. The number of attention map is 32.）
- 2、WSDAN的attention map引导的数据增强有crop和drop。MA提出mix-up（大于阈值的区域记为1，画bounding box，resize成原图大小，两者按照一定比例相加合并、patching（复制）、dimming（类似drop）
- 3、loss。WSDAN的attention regularization loss修改为Soft Distance Regularization using P, p 1, p 2, p 3to calculate overall loss。

## 摘要

任务：利用CXR图像将COVID-19从肺炎病例中分类。

挑战：shared spatial characteristics, high feature variation and contrast diversity between cases. 病例之间具有共同的空间特征、高特征变化和对比度多样性。以及数据少。Moreover, massive data collection is impractical for a newly emerged disease, which limited the performance of data thirsty deep learning models. 

关于名字：Multiscale Attention Guided deep network with Soft Distance regularization (MAG-SD) is proposed to automatically classify COVID-19 from pneumonia CXR images. 

在MAG-SD中，MA-Net被用来从多尺度特征图中产生预测向量和注意力。为了提高训练模型的鲁棒性和缓解训练数据的不足，提出了注意力引导的增强和软距离正则化，其目的是产生有意义的增强并减少噪音。

## 介绍

关于任务的医学背景，这个任务和人工智能辅助诊疗的可能性，以及现有的工作，数据集的来源。

**存在的问题：**一般来说，目前在CXR图像上操作的研究大多依赖于在线数据集和有限的COVID-19病例。不足的数据难以评估模型的稳健性，并限制了其普遍性。在极其不平衡的数据集上训练的模型也导致了长尾分布问题。尽管有很多作品讨论了通过人工智能诊断COVID-19的问题，但由于几个问题，很少有作品解决**不平衡数据和数据集的有限规模问题**。1）由不平衡数据训练的模型倾向于将所有目标分类到主导类，而主导类的标签绝大多数都比其他类多。2）X射线图像上的独特标签，如L/R位置标签，很容易引起模型的注意，然后误导预测。3）COVID-19病例与非COVID病例有共同的特征，这就要求有一个敏感和强大的模型来进行分类。

## 三个贡献

1）设计了一个新的深度网络，MA-Net，将COVID19的诊断作为一个FGVC问题。引入了多尺度注意来评估多层次特征的注意图。组成的注意图被用作训练步骤的指导。提出了注意力集合，以利用注意力图进行分类。

2）通过提出注意力引导的数据增强和多镜头训练阶段来解决数据短缺问题。它包括注意混合、注意修补和注意调光，可以增强和搜索局部特征，然后生成数据。模型在不平衡的COVID-19数据集上进行了训练，并达到了最先进的水平。

3）在不引入其他模块或参数的情况下，制定了一个新的正则化术语，利用预测之间的软距离，作为一个约束条件，限制分类器对一个目标产生矛盾的输出。

## related work

肺炎X射线图像、细粒度的视觉分类、CNN的注意机制和计算机视觉中利用的多尺度特征融合。

多尺度特征融合：从多分辨率的输入图像中提取混合特征图是自手工设计特征的时代以来计算机视觉中的一个常见策略。CNN具有固有的金字塔形状的多尺度特征，如果进行有效的特征融合，在产生强大的语义表征方面是很有利的。U-Net[37]和V-Net[38]等模型利用了跳过连接来关联不同分辨率的特征图。FPN[39]利用多尺度层次的预测，产生了多个预测。对于CXR图像，Huang等人[40]提出了权重串联的方法来合作全局和局部特征。空间注意力的蓬勃发展给了人们从多分辨率特征图中提取注意力的灵感。Sedai等人[41]提出了用于胸部病变定位的A-CNN，该方法通过计算特征图的加权平均数进行多尺度的注意。

