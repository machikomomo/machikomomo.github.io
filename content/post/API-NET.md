---
author: "momo"
date: 2022-05-17
title: "API-NET论文笔记"
categories: [
    "论文笔记",
    "医学图像分析",
    "AAAI",
]

---

## 我的启发

1、图像成对输入的写法。2、两个不同图像的一种交互方式。3、生成注意力的计算方法。

## 引用

Learning Attentive Pairwise Interaction for Fine-Grained Classification,

AAAI, 2020,

https://arxiv.org/abs/2002.10191.


## 博客

https://mp.weixin.qq.com/s/RrMxbnoTPtbHKduNPIvI8g

https://blog.csdn.net/qq_34317565/article/details/108028839?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22108028839%22%2C%22source%22%3A%22momoka9%22%7D&ctrtid=bXigX

## 简介

Attentive Pairwise Interaction Network (API-Net)，Attentive对应的是注意力机制，Pairwise表明训练用的是成对训练的方式，Interaction对应的是交互机制。

网络结构如下。首先将数据两两成组，（两个不同的类别）两张图分别通过一个CNN网络得到特征向量，x1和x2，再将x1和x2拼接在一起，通过一个多层感知器（Multilayer Perceptron，MLP）得到共同向量xm。

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/2022-05-197.31.49.png)

**我们期望xm标记了x1和x2差距大的位置，x1和x2不一致的地方的值接近为1，相同的地方的值接近0。**打个比方，如果x1，x2向量第一个位置标记的是鸟爪的颜色，如果两只鸟鸟爪的颜色很不一样，那么我们期望xm第一个位置的值接近1，如果两只鸟鸟爪的颜色一致，那么我们期望xm第一个位置的值接近0。当然每个位置代表的特征不是事先定义好的，这里只是举个例子，具象化解释一下。



接下来分别计算两个特征向量的注意向量g1和g2（门向量生成）：

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/2022-05-197.31.59.png)

g1被期望表达的是x1和x2不同的地方，且这个不同的地方是x1的主要特征，而g2表达的是x2与x1不同的地方，且这个不同的地方是x2的主要特征（这里体现注意力）。比如说向量的一个位置表达的是爪子是否是黄色，x1在这个位置值高，x2在这个位置值低，那么我们就期望g1在这个位置的值会去接近1，g2在这个位置的值会去接近0。那么如果x1和x2在这个位置的值都高和都低呢？那么就期望g1和g2在这个位置的值都低。

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/2022-05-19_1.49.30.png)

接下来说**Pairwise Interaction**。self可以理解为强化自己的重点，other可以理解为强化别人的重点。

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/2022-05-197.33.03.png)

softmax做最后的分类归纳。

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/2022-05-197.33.07.png)

最后的损失函数：

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/2022-05-197.34.04.png)

前半部分用的是交叉熵，表达的是两个分布之间的距离，也可以理解为预测的结果和真实的结果之间的距离。

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/2022-05-197.34.08.png)

后半部分用的是Score Ranking Regularization，这部分想要表达的是，强化了自己的重点的向量得到的分类结果，应该要比强化了别人的重点得到的分类结果要更准确。（这个是作者的创新，但是没看懂，据说引入了铰链损失函数hing loss）

## 代码（pytorch）

https://github.com/PeiqinZhuang/API-Net

