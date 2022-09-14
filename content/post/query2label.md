---
author: "momo"
date: 2022-09-05
title: "Query2Label"
categories: [
    "论文笔记",
]
---

## Query2Label

论文及代码复现

论文：https://arxiv.org/pdf/2107.10834.pdf

博客：https://zhuanlan.zhihu.com/p/470104101

官方代码：https://github.com/SlongLiu/query2labels

其他人的代码（相对官方的更简洁一点，可能可以实现差不多的SOTA效果）：https://github.com/curt-tigges/query2label

数据集：MS-COCO



## MS-COCO

简介：https://zhuanlan.zhihu.com/p/32566503

自然图像数据集，可以用来做多标签分类和其他任务，比较大，本地下载也没什么意义。

主要看一下数据集（需要输入的）是怎么做的。



## Dataset

```python
from q2l_labeller.data.coco_data_module import COCODataModule

coco = COCODataModule(
    img_data_dir,
    img_size=param_dict["image_dim"],
    batch_size=param_dict["batch_size"],
    num_workers=24,
    use_cutmix=param_dict["use_cutmix"],
    cutmix_alpha=1.0) # 实例化一个数据集以后，放到参数dict中
    
param_dict["data"] = coco
```

主要是看一下这个DataModule返回的是什么。一般来讲就是返回image和label。这里是可以通过调用函数，返回DataLoader。



## model——timm

timm加载预训练模型的差异

```python
model = timm.create_model('resnet50', pretrained=True)
'''
  (global_pool): SelectAdaptivePool2d (pool_type=avg, flatten=Flatten(start_dim=1, end_dim=-1))
  (fc): Linear(in_features=2048, out_features=1000, bias=True)
'''
model2 = timm.create_model('resnet50', pretrained=True, num_classes=0, global_pool="")
'''
  (global_pool): SelectAdaptivePool2d (pool_type=, flatten=Identity())
  (fc): Identity()
'''
```

nn.Identity()模块不改变输入：https://blog.csdn.net/weixin_43135178/article/details/118710051

所以model2表示删除最后的池化层和全连接层，即只保留特征提取的backbone。



## model——nn.Transformer输入

```
49,12,256 # 图像特征
14,12,256  # 标签的embedding
后两个维度的数字是一样的
```

https://zhuanlan.zhihu.com/p/107586681



## 训练过程

```python
pl_model = Query2LabelTrainModule(**param_dict)
```

```python
import pytorch_lightning as pl

trainer = pl.Trainer(
    max_epochs=24,
    precision=16,
    accelerator='gpu', 
    devices=1,
    logger=wandb_logger, # Comment out if not using wandb
    default_root_dir="training/checkpoints/",
    callbacks=[TQDMProgressBar(refresh_rate=10)])

trainer.fit(pl_model, param_dict["data"])
这写法也真的太奇怪了……主要还是用了pytorch_lightning的东西
```



```python
class Query2LabelTrainModule(pl.LightningModule):
```

这个是继承着写的，所以可以套用。



loss

```
loss = self.base_criterion(y_hat, y.type(torch.float))
```



## 介绍

The use of Transformer is rooted in the need of extracting local discriminative features adaptively for different labels, which is a strongly desired property due to the existence of multiple objects in one image.一张图片里可能有多个对象/标签，所以需要对不同的标签自适应地提取局部区别性的特征。

The built-in cross-attention module in the Transformer decoder offers an effective way to use label embeddings as queries to probe and pool class-related features from a feature map computed by a vision backbone for subsequent binary classifications.

使用的是transformer解码器里面交叉注意力的模块。



## 位置编码

https://blog.csdn.net/m0_37052320/article/details/103979358

https://blog.csdn.net/weixin_44966641/article/details/119299678

本文没用上啊。



## transformer的pytorch实现

https://github.com/xmu-xiaoma666/External-Attention-pytorch
