---
author: "momo"
date: 2022-07-27
title: "odor-5k-0"
categories: [
    "论文笔记",
]
---

## AAAI2022

**[Inferring Prototypes for Multi-Label Few-Shot Image Classification with Word Vector Guided Attention](https://www.x-mol.com/paperRedirect/1466853195513372672)**

（1.标签转为词向量）

多标签图像分类（ML-IC）+少量图像分类（FSIC）=多标签少量图像分类模型（ML-FSIC）multi-label+few-shot

动机：不同的标签往往指的是图像的不同部分。例如，给定一个描述汽车和自行车的图像，使用整个图像的表示来获得自行车的原型将是误导的。所以需要一个基于局部图像特征的策略，使我们能够专注于训练图像中最有可能相关的部分。

然而，由于我们可能只有一些标签的单一训练例子，如果没有某种关于标签含义的先验知识，我们就无法实施这样的策略。为此，我们将依靠词向量（Pennington, Socher, and Manning 2014）。之前一些针对单标签设置的工作已经依靠词向量来直接推断原型（Xing等人，2019；Yan等人，2021a），但由于产生的原型不可避免地有噪音，这种策略在与来自视觉特征的原型结合时最有用。——我的理解：把标签本身的语义作为特征和之前提取的特征图作融合。

在本文中我们只使用词向量来识别训练图像中哪些区域最有可能与给定的标签相关。13个标签，标签名确实有这方面提示。角膜、结膜。那么这个可不可以做到呢？仅仅依靠词向量就能识别训练图像中哪个区域和当前标签最相关？

假设我们有一些指代动物的标签。这些标签会有类似的词向量，这就告诉模型，这些不同标签的预测性视觉特征可能是相似的。

**关键点1:同类/同个区域的视觉特征应当有类似的词向量。**

现在，假设我们有一张被标记为猫的图像。根据其他标签的训练数据，该模型将选择可能包含动物的区域（尽管它不一定能够区分猫和密切相关的动物）。

图像中提取特征-局部特征-全局平均池化-全局特征-投影

预训练的词向量-投影

两者共同得到一个joint embedding space。联合embedding空间。这个组件只是用来学习视觉特征和标签的联合embedding。由于损失L cmw的目的是将两种不同的模式（词向量和视觉特征）对齐，我们将其称为跨模式权重损失（CMW-loss）。

词向量的获取：具体来说，我们使用了从维基百科2014和Gigaword 5中训练出来的向量，这些向量是我们从GloVe项目页面上获得的，网址是https://nlp.stanford.edu/projects/glove。

![截屏2022-07-27 下午12.26.47](/Users/momochan/Library/Application Support/typora-user-images/截屏2022-07-27 下午12.26.47.png)





注意力机制：qkv。标签embedding作为query。

![截屏2022-07-27 下午12.30.51](/Users/momochan/Library/Application Support/typora-user-images/截屏2022-07-27 下午12.30.51.png)



query images？

![截屏2022-07-27 下午12.35.52](/Users/momochan/Library/Application Support/typora-user-images/截屏2022-07-27 下午12.35.52.png)

## 标签-词向量

13种病变标签：wikipedia（部分找不到，deepl翻译）



白内障 cataract

人工晶体 Intraocular lens

晶状体脱位 Ectopia lentis -> https://eyewiki.aao.org/Ectopia_Lentis

角膜炎 Keratitis

角膜瘢痕 Corneal scarring

角膜变性 Corneal dystrophy（角膜营养不良）

角结膜肿瘤 Corneal Conjunctival Tumor

睑裂斑 Palpebral fissure

翼状胬肉 pterygium

结膜下出血 subconjunctival hemorrhage

结膜充血 Conjunctival congestion

结膜囊肿 Conjunctival cysts

色素痣 Pigmented nevus

原文，采用Wikipedia 2014 + Gigaword 5（6B 令牌，400K 词汇，无大小写，300d 向量，822 MB 下载）：[glove.6B.zip](https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip) [ [mirror](https://nlp.stanford.edu/data/wordvecs/glove.6B.zip) ]预训练的词向量。

怎么用呢？

以及，怎么去

https://blog.csdn.net/nlpuser/article/details/83627709?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165889932016782184612764%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=165889932016782184612764&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~pc_rank_34-5-83627709-null-null.142^v35^pc_rank_34,185^v2^control&utm_term=%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%8D%E5%90%91%E9%87%8F&spm=1018.2226.3001.4187

就这样吧……懒得搞了。

之前说的dynamic mlp：

https://github.com/ylingfeng/DynamicMLP

shift 操作：

https://www.msra.cn/zh-cn/news/features/aaai-2022



1.标签的语义和文件名语义和图片语义的结合，弱监督的图像定位，借助标签去对图像关注区域做大致定位

2.标签的类不均衡问题

3.应该就是一个标签对应一个词向量吧？



## Few Shot

1是基于表示的原型网络，匹配网络这种就是，学到了不同的表示再进行对比，你就可以理解为metric based

2是基于策略的，比如maml这种

这个可以随便找一篇看看related works..

https://blog.csdn.net/m0_38031488/article/details/85274890

https://blog.csdn.net/u014767662/article/details/81670215



support set 很小的数据集，带标签（不足以训练大的神经网络）

query 拿query和support set依次对比，最接近最相似的

meta learning 看作一个东西

lear to learn

一张就是 one shot

training set，support set，query

k-way：support set里面有k个类别（样本均衡的情况下，随着类别数量增加，acc会降低）

n-shot：每个class有n个样本（样本均衡的情况下，随着每个类别样本数量增加，acc会升高）



basic idea：1.从一个很大的训练集上学习相似度，可以判断两张图片的相似度（比如imagenet）

2.将query和support set里面的每一张图片逐一做对比，计算相似度



omniglot（1600个类 20个样本） 类似mnist（手写数字识别，10 6000）

50个字母表（不同语言），每个字母表有很多字符，1623个字符。

每个字符有20个人手写，所以有20个样本。

![截屏2022-08-12 下午12.38.00](/Users/momochan/Library/Application Support/typora-user-images/截屏2022-08-12 下午12.38.00.png)



mini imagenet

100个类别 每个类别600个样本 = 60000

84*84的小图片

https://zhuanlan.zhihu.com/p/437414450



训练孪生网络的两种方式：

1.pairwise



![截屏2022-08-12 下午5.53.25](/Users/momochan/Library/Application Support/typora-user-images/截屏2022-08-12 下午5.53.25.png)



2.triplet loss

![截屏2022-08-12 下午5.54.06](/Users/momochan/Library/Application Support/typora-user-images/截屏2022-08-12 下午5.54.06.png)

3.训练完孪生网络以后，可以提取特征，然后比较两张图片，映射成特征向量，比较距离。

few-shot 的另一种方法：

在大的数据集上预训练网络，将query和support set在特征空间上的特征向量，拿来做相似度比较（比如cos）

更具体的：把均值向量做归一化，得到归一向量，其二范数（向量元素绝对值的平方和再开方）均为1.

![截屏2022-08-12 下午6.10.26](/Users/momochan/Library/Application Support/typora-user-images/截屏2022-08-12 下午6.10.26.png)



做预测：先把上面得到的表征堆叠起来得到矩阵M，再用query对应的特征向量q和M作Mq，再做softmax。



![截屏2022-08-12 下午6.14.09](/Users/momochan/截屏/截屏2022-08-12 下午6.14.09.png)

![截屏2022-08-12 下午6.16.05](/Users/momochan/截屏/截屏2022-08-12 下午6.16.05.png)

改进：fine-tune 在support set上进行微调（利用support set再训练一个小的分类器）（技巧，把分类器的W初始化为M，b初始化为0）

![截屏2022-08-12 下午6.22.25](/Users/momochan/截屏/截屏2022-08-12 下午6.22.25.png)



再训练小分类器的时候，使用entropy regularization

![截屏2022-08-12 下午6.27.15](/Users/momochan/截屏/截屏2022-08-12 下午6.27.15.png)





Trick3，把softmax里面的第一项，修改一下。改成cos。

![截屏2022-08-12 下午6.28.21](/Users/momochan/截屏/截屏2022-08-12 下午6.28.21.png)
