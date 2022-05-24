---
author: "momo"
date: 2022-05-23
title: "luna论文笔记"
categories: [
    "论文笔记",
    "AAAI",
]
---

# LUNA: Localizing Unfamiliarity Near Acquaintance for Open-set Long-Tailed Recognition

https://www.aaai.org/AAAI22Papers/AAAI-10200.CaiJ.pdf

2022 AAAI

## 博客

center loss：

https://blog.csdn.net/duan19920101/article/details/104445423

https://github.com/KaiyangZhou/pytorch-center-loss （center_loss.py）

## 动机

However, the performances of the state-of-the-art object recognition methods mostly bias on the sample-rich classes that have been seen in the training set, with a limited ability on classifying the sample-few classes, not to mention the new/novel classes of objects (Kang et al. 2019; Zhou et al. 2020).分类表现倾向于那些样本丰富的类别，对于那些样本量比较少的类别，分类能力有限。两个问题：open-set和长尾。两个挑战往往是重合的。Open-set Long-Tailed Recognition简称OLTR。

作者提出一个度量学习框架，称为Localizing Unfamiliarity Near Acquaintance（LUNA），根据深度CNN特征的局部密度（local density）来定量测量开放集长尾识别任务的新颖程度。通过LUNA，可以精确地回答两个问题：（1）输入是否是新的；（2）如果不是，是哪一类；if yes, what is the unfamiliarity level of the new class concerning the pretrained acquaintance classes. 如果是新类，这个新类和预训练好的类别的不相似等级是多少。综上所述，我们声称我们的贡献和技术革新如下。

1、我们收集了一个新的注释良好的真实海洋物种开放长尾（MS-LT）数据集。作为细粒度领域的第一个自然OLTR数据集，它将是对现有人工重新采样的OLTR数据集的有力补充。它对表征学习和新物种检测提出了新的挑战。

2、为了使类别单独集中在特征空间中，特征提取器通过新提出的损失，即加权中心损失**（wcenter-loss）**来训练，以最小化它们的类内距离，从而在高维空间中形成密集的聚类。它集中了头类的深层特征，同时保留了尾类的分类精度，从而使特征更加鲜明。

3、为了衡量新类的不熟悉程度，评价与熟人类的接近程度，我们提出了一个LUNA因子，一个基于深层特征的相对密度的离群指标，它对分布是自适应的。LUNA因子是第一个对长尾分布下的新颖性进行定量测量的指标。

4、我们在MS-LT数据集和两个常用的人工采样数据集ImageNet-LT和Place-LT上对LUNA进行了广泛的评估，包括长尾和Openet识别。结果表明，LUNA在封闭集上明显优于最先进的方法4-6%，在开放集设置下，F-measure平均提高4%。

## related work

1.OLTR

2.Novelty Detection

3.深度度量学习（DML）。DML是在高维嵌入空间中最大化类间距离和最小化类内距离。两种类型的DML方法被广泛使用：a）用类级标签学习；b）图像级标签。前者从分类模型中获得嵌入，例如ArcFace（Deng等人，2019），CosFace（Wang等人，2018）。后者通过损失函数直接优化采样图像对或组的嵌入距离，不产生DML后的分类结果，如对比性（Chopra, Hadsell, and LeCun 2005）、三联体（Schroff, Kalenichenko, and Philbin 2015）和中心（Wen et al. 2016）损失。这些DML算法都是在训练数据充足且普遍平衡的假设下进行的，这对于长尾设置来说并不成立。对于类级的DML，分类精度主要受数据分布偏差的影响；而对于图像级的DML，少数照片的类容易被过度拟合。本文提出了一个频率感知的损失函数来同时解决数据不平衡和度量学习问题。

## attention regularization loss

f和c各自做L2_norm。作差，平方，sum，mean。

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/2022-05-214.37.10.png)

官方实现。

```
def calculate_pooling_center_loss(features, label, alfa, nrof_classes, weights, name):
    features = tf.reshape(features, [features.shape[0], -1])
    label = tf.argmax(label, 1)

    nrof_features = features.get_shape()[1]
    centers = tf.get_variable(name, [nrof_classes, nrof_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    centers_batch = tf.nn.l2_normalize(centers_batch, axis=-1)

    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)

    with tf.control_dependencies([centers]):
        distance = tf.square(features - centers_batch)
        distance = tf.reduce_sum(distance, axis=-1)
        center_loss = tf.reduce_mean(distance)

    center_loss = tf.identity(center_loss * weights, name=name + '_loss')
    return center_loss
```

训练集里面，一个类别中的样本越少，它的权重就越高。取样本数量的倒数？作为lambda加进去？（感觉太粗暴了，但是应该可行吧）

## center loss

Cyi表示第yi个类别的特征中心，Xi表示全连接层之前的特征。实际使用的时候，m表示mini-batch的大小。因此这个公式就是希望一个batch中的每个样本的feature离feature 的中心的距离的平方和要越小越好，也就是类内距离要越小越好。

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/2022-05-237.49.52.png)



![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/2022-05-237.26.03.png)

目标函数如下：

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/2022-05-237.34.53.png)

## wcenter loss（加权的center loss）

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/2022-05-237.35.53.png)

在长尾数据集的情况下，由于样本数量少得多，尾部类的分布往往比较稀疏，更容易与特征空间中的其他聚类混在一起。因此，我们提出了一个加权中心（wcenter）损失，以适应不平衡的分布。

c是类别的索引。λi 是 yi所属的类j的归一化频率的权重。（黑人问号脸）

nj表示训练集中j类样本的数量。

 基本上，λi是由最大频率值缩放的反转分布，表示为nj波浪，然后在[0，1]之间归一化。

一个类别中的样本越少，它的权重就越高。

请注意，λi应大于1，并处于交叉熵损失的同一尺度，以确保网络的收敛性。

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/2022-05-237.26.10.png)

目标函数改成下图，请注意，参数λ被嵌入到Lwc中，并为不同的类定制，以最小化类内距离，特别是尾部。

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/2022-05-237.33.57.png)

