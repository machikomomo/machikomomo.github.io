---
author: "momo"
date: 2022-06-30
title: "tricks"
categories: [
    "论文笔记",
]
---

## 数据增强

#### 二元关系法

将每个标签看作是一个独立的问题，即建立多个分类器，每个分类器只针对一种标签进行二分类。这种方法相对简单，适用性广。

1.对标签数量很多的情况，需要对每个标签构造一个分类模型。

2.可能存在类别不均衡问题。

3.所有标签独立处理，没有考虑到标签相关性。

#### 分类器链

选取一个排序函数，将标签按照排序函数进行排列，前一个分类器的标签当作下一个分类器的输入，以链条的形式组合起来。第一个分类器的输入为原始输入X；第二个分类器的输入为原始输入X和y1标签……

1.适用于标签之间存在关联或者包含关系的时候。

2.前一个分类结果如果错误，会影响后面的分类准确性。

3.本质上是二分类，但是考虑了标签相关性。

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/12611656314163_.pic.jpg)

#### 标签集

将多个标签组合起来，把具有多个相同标签的样本标记为一个超集，这样把原始问题转化为一个多分类的问题。

1.类别不均衡。如果标签的组合过多，每个超集的样本数量就会太少，不能达到训练分类器的要求。

2.测试阶段只能预测得到训练时出现过的超集，不能预测得到新的标签组合。

## 算法适应法

#### CNN-RNN（2017 CVPR）

首先输入一张图片，经过VGG网络对图片特征进行提取，然后并将图片特征嵌入到联合embedding空间，同样将标签也嵌入到此空间当中，红点代表标签embedding，蓝点代表图像embedding，黑点则表示图像embedding和递归神经元输出embedding的总和。

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/12621656316686_.pic.jpg)

1.为了分析标签之间的高阶关系，使用LSTM作为RNN神经元。
2.使用CNN对图像进行特征提取，使用RNN对多标签进行编码。
3.训练时，网络输入为图像和标签，网络输出为标签。
4.预测时，使用束搜索对预测序列进行判断，选择最优解。每次搜索过程中，使用CNN进行特征提取，将特征和当前的标签结合，作为预测层的输入，得到输出序列，然后进行下一步地搜索。由原文公式(4)，提取的特征和标签先通过线性变换，相加，再投影到预测层，因此第一次预测的时候不需要标签，之后的每一次预测都用到前一次的预测结果。

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/20220627160530.png)

## 多示例学习

#### Attention based mil

Ilse M, Tomczak J, Welling M. Attention-based deep multiple instance learning[C]//International conference on machine learning. PMLR, 2018: 2127-2136.

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/12821656586407_.pic.jpg)

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/12801656586216_.pic.jpg)



## 医学图像分析领域的多标签数据集

#### ODIR-5k

简介：这是一个由5,000名病人组成的结构化眼科数据集。提供了八种眼病类别的多标签图像级注释，包括糖尿病、青光眼、白内障、老年黄斑变性（AMD）、高血压、近视、正常和其他疾病。每个病人可能包含一个或多个疾病标签。模型需要开发自动眼部疾病分类的方法。以左右眼的彩色眼底图像作为输入（可以使用其他提供的信息，如患者年龄、性别），目标是将患者分为八类。

原比赛：https://odir2019.grand-challenge.org/

参与者必须提交所有测试数据的八类分类结果。对于每个类别，分类概率（值从 0.0 到 1.0）表示被诊断为相应类别的患者的风险。 提交的内容根据三个指标进行评分：[kappa score ](https://en.wikipedia.org/wiki/Cohen's_kappa)、  [F-1 socre](https://en.wikipedia.org/wiki/F1_score) 和 [AUC value](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve)。 阈值为 0.5。

数据集下载地址：https://www.icode9.com/content-2-869077.html

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/12791656585757_.pic.jpg)

相关论文：

1.https://arxiv.org/abs/2008.09772  A Benchmark for Studying Diabetic Retinopathy: Segmentation, Grading, and Transferability -- IEEE Transactions on Medical Imaging (2020)

2.https://www.jmir.org/2021/7/e27822/ 

#### FGADR

包含2,842张图像的细粒度注释DR（糖尿病视网膜病变）数据集（FGADR）。这个数据集有1,842张带有像素级DR相关病变注释的图像，以及1,000张带有图像级标签的图像，建立了三个基准任务进行评估。1. DR病变的分割；2.通过联合分类和分割对DR进行分级；3.用于眼部多病种识别的转移学习。此外，我们还为第三个任务引入了一种新的归纳转移学习方法。

https://csyizhou.github.io/FGADR/

相关论文：https://arxiv.org/abs/2008.09772  A Benchmark for Studying Diabetic Retinopathy: Segmentation, Grading, and Transferability -- IEEE Transactions on Medical Imaging (2020)

为了评估眼部多病种识别，从ODIR-5K[31]数据集中的7000张图像被用于训练和验证。进行了五重交叉验证实验。表五显示了不同方法的结果。
我们首先评估单个模型，VGG-16、Inception v3和我们的DenseNet架构，作为基线，其中DenseNet取得了最好的性能。然后，在源域任务学习的帮助下，多尺度转移连接
(MTC)将Kappa提高了2.87%。此外，特定领域的对抗性适应（DSAA）模块可以进一步提高模型性能，使Kappa增加5.05%。两种设计的有效性都得到了验证。与对两个领域采用相同的BN层的普通对抗性适应（AA）相比，特定领域判别器的独立BN层使Kappa增加了1.96%。对于更多的细节，每种疾病的分类准确率在表六中得到说明。我们观察到，从我们的细粒度注释的DR领域数据中进行的迁移学习可以持续改善任务域中所有眼部疾病的识别结果。特别是，对于糖尿病、AMD和高血压，改进是显著的，而对于青光眼、白内障和近视，则取得了轻微的收益。为了更好地解释从源域到目标域的迁移学习的有效性，我们将被我们的迁移学习方法正确分类但被错误分类的样本的最终Logit图可视化。

通过www.DeepL.com/Translator（免费版）翻译
