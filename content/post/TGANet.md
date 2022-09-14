---
author: "momo"
date: 2022-08-23
title: "TGANet"
categories: [
    "论文笔记",
]
---

## TGANet:Text-guided attention for improved polyp segmentation

论文地址：https://arxiv.org/abs/2205.04280

github地址：https://github.com/nikhilroxtomar/tganet

MICCAI 2022

所提出的TGANet是一种具有文本引导的注意力机制的息肉分割架构，能够增强特征表示，使得图像中存在的息肉被最佳分割，而无论息肉的大小变化和形状。

本文工作的主要贡献包括：

1）文本引导注意力，在息肉数量（一个或多个）和大小（小、中、大）的背景下学习不同的特征；

2）特征增强模型，以增强编码器的特征并将其传递给解码器；

3）多尺度特征融合，以捕获不同类型的息肉所学习的特征



## 网络架构（a）

（a）中的编码结构，采用resnet50作为backbone，block1-block4既用于“辅助属性分类任务”（auxiliary attribute classification task），又用于“主干息肉分割任务”（main polyp segmentation task）。

“辅助属性分类任务”：采用block4的输出，加softmax，得到两个分类结果（息肉数量，息肉大小）

“主干息肉分割任务”：4个block的输出（feature maps）经过FEM（特征增强模块）得到新的特征（通过膨胀卷积和注意力机制）



## 网络架构（b）FEM

特征增强模块（FEM）从四个平行的膨胀卷积Conv开始，膨胀率为{1,6,12,18}。每个膨胀之后是一个批量归一化BN和一个校正的线性单元RELU，我们称之为CBR。输出特征通过通道注意模块CAM[17]来捕获通道特征之间的显式关系。然后，将这四个扩展卷积的突出特征串接并通过Conv3×3，然后通过BN层，并和“原始特征通过Conv1×1和BN得到的特征“相加。随后是一个ReLU激活函数，并应用空间注意力机制SAM[17]来抑制不相关区域。



## 网络架构（c）Label Attention

3个label attention模块——给代表性特征更高权重，并抑制冗余特征。

softmax结果，concat——>

辅助信息，使用byte-pair编码，进行embedding——>

上面两个经过元素点积得到一个结果。经过label attention，残差连接，得到l1.

l1经过label attention，残差连接，得到l2

l2经过label attention，得到l3

每次直接经过attention的结果就去做d运算。（decoder）



## 我的理解

准备数据：原始数据是：**<u>图片</u>**（16，3，256，256）和**<u>mask</u>**（16，3，256，256）

1、根据mask，获得息肉区域的bbox，再根据bbox获得**<u>息肉的数量</u>**（16，2）以及<u>**息肉大小**</u>（16，3）

2、**<u>辅助信息</u>**，始终是固定的，是一个（5，300）的tensor

制作数据集：如上五类数据。数据：图片+辅助信息；标签：mask+息肉数量+息肉大小

前向计算：1.图片经过backbone提取，预测得到息肉数量、息肉大小；2.预测得到的这两个值，与辅助信息进行融合，帮助分割，推测得到mask。

loss计算：[DiceBCELoss(), nn.CrossEntropyLoss(), nn.CrossEntropyLoss()] 三个loss。平均作为最终loss。



## 我的启发

准备数据：原始数据是：**<u>图片</u>**（16，3，256，256）和**<u>类别标签</u>**

1、根据类别标签，可以获得<u>**区域标签**</u>

2、**<u>辅助信息</u>**，始终是固定的，是一个（（瞳孔、结膜、角膜）300）的tensor

制作数据集：如上四类数据。数据：图片+辅助信息；标签：类别标签+区域标签

前向计算：1.图片经过backbone提取，预测得到区域标签；2.预测得到的区域标签值，与后面的特征提取模块融合，帮助分类，推测得到最终的类别标签。

loss计算：[BCELoss, BCELoss] 两个loss。平均（动态分配）作为最终loss。

## 

![截屏2022-08-23 上午10.19.22](/Users/momochan/Library/Application Support/typora-user-images/截屏2022-08-23 上午10.19.22.png)



## 网络架构（d）MSFA——多尺度特征融合模块

![截屏2022-08-23 下午12.45.00](/Users/momochan/Library/Application Support/typora-user-images/截屏2022-08-23 下午12.45.00.png)

## byte pair encoding

https://blog.csdn.net/m0_37962192/article/details/117417537

https://anaconda.org/powerai/bpemb



## result

5个数据集

## ![截屏2022-08-25 下午6.50.35](/Users/momochan/Library/Application Support/typora-user-images/截屏2022-08-25 下午6.50.35.png)
