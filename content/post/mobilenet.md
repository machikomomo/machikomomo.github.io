---
author: "momo"
date: 2022-05-12
title: "mobile net"
categories: [
    "论文笔记",
]

---

## MobileNet V1

MobileNets是为移动和嵌入式设备提出的高效模型。MobileNets基于流线型架构(streamlined)，使用深度可分离卷积(depthwise separable convolutions,即Xception变体结构)来构建轻量级深度神经网络。允许通过两个超参数直接构建非常小、低延迟、易满足嵌入式设备要求的模型。

论文介绍了两个简单的全局超参数，可有效地在延迟和准确率之间做折中。这些超参数允许我们依据约束条件选择合适大小的模型。论文测试在多个参数量下做了广泛的实验，并在ImageNet分类任务上与其他先进模型做了对比，显示了强大的性能。论文验证了模型在其他领域(对象检测，人脸识别，大规模地理定位等)使用的有效性。

## 引用

Learning Attentive Pairwise Interaction for Fine-Grained Classification,

AAAI, 2020,

https://arxiv.org/abs/2002.10191.


## Related Work

在建立小型高效的神经网络工作中，通常可分为两类工作：

1、压缩预训练模型。获得小型网络的一个办法是减小、分解或压缩预训练网络，例如量化压缩(product quantization)、哈希(hashing )、剪枝(pruning)、矢量编码( vector quantization)和霍夫曼编码(Huffman coding)等；此外还有各种分解因子(various factorizations )用来加速预训练网络；还有一种训练小型网络的方法叫蒸馏(distillation )，使用大型网络指导小型网络，这是对论文的方法做了一个补充，后续有介绍补充。

2、直接训练小型模型。 例如Flattened networks利用完全的因式分解的卷积网络构建模型，显示出完全分解网络的潜力；Factorized Networks引入了类似的分解卷积以及拓扑连接的使用；Xception network显示了如何扩展深度可分离卷积到Inception V3 networks；Squeezenet 使用一个bottleneck用于构建小型网络。

作者提出的MobileNet网络架构，允许模型开发人员专门选择与其资源限制(延迟、大小)匹配的小型模型，MobileNets主要注重于优化延迟同时考虑小型网络，从深度可分离卷积的角度重新构建模型。

## Depthwise Separable Convolution

MobileNet是基于深度可分离卷积的。通俗的来说，深度可分离卷积干的活是：把标准卷积分解成深度卷积(depthwise convolution)和逐点卷积(pointwise convolution)。这么做的好处是可以大幅度降低参数量和计算量。

对于深度分离卷积，把标准卷积( 4 , 4 , 3 , 5 ) (4,4,3,5)(4,4,3,5)分解为：

深度卷积部分：大小为( 4 , 4 , 1 , 3 ) (4,4,1,3)(4,4,1,3)，作用在输入的每个通道上，输出特征映射为( 3 , 3 , 3 ) (3,3,3)(3,3,3)
逐点卷积部分：大小为( 1 , 1 , 3 , 5 ) (1,1,3,5)(1,1,3,5)，作用在深度卷积的输出特征映射上，得到最终输出为( 3 , 3 , 5 ) (3,3,5)(3,3,5)

MobileNet使用可分离卷积减少了8到9倍的计算量，只损失了一点准确度。

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/2022-05-19_2.29.53.png)



## MobileNet V2

MobileNetV1网络主要思路就是深度可分离卷积的堆叠。在V2的网络设计中，我们除了继续使用深度可分离（中间那个）结构之外，还使用了Expansion layer和 Projection layer。这个projection layer也是使用 ![[公式]](https://www.zhihu.com/equation?tex=1%5Ctimes+1) 的网络结构，他的目的是希望把高维特征映射到低维空间去。另外说一句，使用 ![[公式]](https://www.zhihu.com/equation?tex=1%5Ctimes+1) 的网络结构将高维空间映射到低纬空间的设计有的时候我们也称之为**Bottleneck layer。**

**Expansion layer**的功能正相反，使用 ![[公式]](https://www.zhihu.com/equation?tex=1%5Ctimes+1) 的网络结构，目的是将低维空间映射到高维空间。这里Expansion有一个超参数是维度扩展几倍。可以根据实际情况来做调整的，默认值是6，也就是扩展6倍。

![img](https://pic1.zhimg.com/80/v2-d17f4a89497899e7293ba81b866b0830_1440w.jpg)

此图更详细的展示了整个模块的结构。我们输入是24维，最后输出也是24维。但这个过程中，我们扩展了6倍，然后应用深度可分离卷积进行处理。整个网络是中间胖，两头窄，像一个纺锤形。**bottleneck residual block（ResNet论文中的）**是中间窄两头胖，在MobileNetV2中正好反了过来，所以，在MobileNetV2的论文中我们称这样的网络结构为**Inverted residuals**。需要注意的是residual connection是在输入和输出的部分进行连接。因为从高维向低维转换，使用ReLU激活函数可能会造成信息丢失或破坏（不使用非线性激活函数）。所以在projection convolution这一部分，我们不再使用ReLU激活函数而是使用线性激活函数。

why？下面谈谈为什么要构造一个这样的网络结构。

我们知道，如果tensor维度越低，卷积层的乘法计算量就越小。那么如果整个网络都是低维的tensor，那么整体计算速度就会很快。

然而，如果只是使用低维的tensor效果并不会好。如果卷积层的过滤器都是使用低维的tensor来提取特征的话，那么就没有办法提取到整体的足够多的信息。所以，如果提取特征数据的话，我们可能更希望有高维的tensor来做这个事情。V2就设计这样一个结构来达到平衡。

![img](https://pic4.zhimg.com/80/v2-0595ba48c058f23b476f2ce7b4663237_1440w.jpg)

先通过Expansion layer来扩展维度，之后在用深度可分离卷积来提取特征，之后使用Projection layer来压缩数据，让网络从新变小。因为Expansion layer和Projection layer都是有可以学习的参数，所以整个网络结构可以学习到如何更好地扩展数据和重新压缩数据。

## MobileNet V3

**MobileNet V1**：提出深度可分离卷积；
**MobileNet V2**：提出反转残差线性瓶颈块；

**MobileNet V2**：参考了三种模型：MobileNetV1的深度可分离卷积、MobileNetV2的具有线性瓶颈的反向残差结构(the inverted residual with linear bottleneck)、MnasNe+SE的自动搜索模型。

引入基于squeeze and excitation结构的轻量级注意力模型(SE)。

网络结构搜索中，结合两种技术：资源受限的NAS（platform-aware NAS）与NetAdapt。
