---
author: "momo"
date: 2022-05-09
title: "SimCLR论文笔记"
categories: [
    "论文笔记",
    "医学图像分析",
    "ICML",
]

---

## 我的启发

了解了这个有名的基于对比学习的自监督框架。它的结构比较简单，也容易实现。在csdn上找了pytorch的实现版本（github上也有，但是代码有些看不懂的，晚点对比看看区别）。它的结构分成两个部分。第一阶段是用没有标签的数据完成自监督预训练，第二阶段是用带有标签的数据进行监督学习，完成整个网络框架。缺点就是batch_size真的很大，需要非常充足的算力。

## 引用

A Simple Framework for Contrastive Learning of Visual Representations,

ICML, 2020,

https://arxiv.org/abs/2002.05709.


## 博客

已经有其他很多博客写过了。我觉得写得也很清楚。这里是我看过的几篇。

https://zhuanlan.zhihu.com/p/258958247

https://zhuanlan.zhihu.com/p/197802321

## 简介

如下图，对一个batch中的每一张图片x，都进行两次数据增强。分别得到xi，xj。通过同一个网络进行特征提取（如resnet50），得到hi，hj（文中称为representation）；然后对hi，hj进行非线性变换（投影），文中用了mlp，得到z。用z来进行contrastive loss的计算。

优化loss。

以上整个过程是无监督的，使用没有标签的数据。

优化迭代结束，对应的representation，就拿来用于下游任务。比如分类任务，就在上一阶段的网络架构上，增加一个简单的全连接层，固定全连接层之前的参数不动，选用一些带有标签的数据，进行监督训练。

要注意的是尽管图中加了一层g层（比如mlp），但是这么做的原因仅仅是辅助，拿计算得到的z来进行loss计算。这样做的效果比直接使用h去进行loss计算结果要好。最终目的是得到好的representation用于下游任务。

![截屏2022-05-09 下午8.35.36](/Users/momochan/Library/Application Support/typora-user-images/截屏2022-05-09 下午8.35.36.png)

## contrastive loss

计算loss。下图中#pairwise similarity对照上图，指的是同一张图片经过两种数据增强得到的两个结果。计算相似度。这个相似度越小越好。

当i和除了i（本身）、j（来自于同一张图片）以外的其他图片进行相似度计算的时候，应当是越大越好。

对照下图中的definel(i,j)，分母应该是用来做归一化，同时loss会朝着分子越小，分母越大的趋势优化。

即同一个数据增强以后的图片越相似（类内距离减小）；不同图片增强以后不相似（类间距离增大）。

![截屏2022-05-09 下午8.53.00](/Users/momochan/Library/Application Support/typora-user-images/截屏2022-05-09 下午8.53.00.png)

搬一个别人的解释。假设输入1是dog、输入2是cat，输入3是tree。经过数据增强以后分别得到1a，1b。2a，2b。3a，3b。

“以楼主的1a,1b为例，假设有（1a,1b），（2a,2b），（3a,3b），分子是（1a,1b）的相似度，分母应该是【（1a,1b）+（1a,2a）+（1a,2b）+（1a,3a）+（1a,3b）】”。

（以上都是看别人的博客，感觉还是乱乱的，自己枚举了一下。假设有3个输入，也就是N=3。k=1，2，3.

x1经过数据增强，记为x1，x2.

x2经过数据增强，记为x3，x4.

x3经过数据增强，记为x5，x6.

取第二组数据为例，i=3，j=4.计算l(3,4).分子是**s(3,4)**.分母是**s(3,1)+s(3,2)+s(3,4)+s(3,5)+s(3,6)**

最终的contrasive loss的定义是如图的L.

也就是N=3时：1/6 *（l(1,2)+l(2,1)+l(3,4)+l(4,3)+l(5,6)+l(6,5))

好吧……略麻烦。

## NT-Xent

补充一下。作者的contrastive loss是基于NT-Xent得到的一个总的优化目标。NT-Xent非原创。

![截屏2022-05-09 下午9.15.40](/Users/momochan/Library/Application Support/typora-user-images/截屏2022-05-09 下午9.16.03.png)

## 如何评价representation的好坏

通过固定特征提取器 ![[公式]](https://www.zhihu.com/equation?tex=f) ，然后加上一个线性分类器输出one-hot。通过监督学习（有label）训练线性分类器，得到的精度作为评估![[公式]](https://www.zhihu.com/equation?tex=f)提取特征能力的指标。

## 三个重要观点

(1) Data augmentation is crucial to UCL；

单纯的图片裁剪没什么效果。加上了color distortion才有显著效果。左边8张图，只有裁剪。看直方图就能分辨是否是一张图片裁剪出来的。右边8张图，加上color distortion。

![img](https://miro.medium.com/max/1400/1*rujTYcDmDRxxpeTT_CmZAw.png)

(2) More variables and training steps give better performance；

模型大，参数多，增加训练时间，效果好。

(3) Large batch size benefit SimCLR。

batch_size越大越好。好贵。

![img](https://miro.medium.com/max/1400/1*yrzj_3xxzWNBWBp8JjgNJw.png)

