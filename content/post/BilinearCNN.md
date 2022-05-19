---
author: "momo"
date: 2022-05-19
title: "BilinearCNN"
categories: [
    "论文笔记",
]

---

## 引用

Bilinear CNN Models for Fine-grained Visual Recognition

## 博客

https://zhuanlan.zhihu.com/p/62532887 （乱）

https://github.com/Iceland-Leo/Bilinear-CNN （非官方pytorch实现）

## 简介

bilinear pooling在2015年于《Bilinear CNN Models for Fine-grained Visual Recognition》被提出来用于fine-grained分类后，又引发了一波关注。bilinear pooling主要用于特征融合，对于从同一个样本提取出来的特征 ![[公式]](https://www.zhihu.com/equation?tex=x) 和特征 ![[公式]](https://www.zhihu.com/equation?tex=y) ，通过bilinear pooling得到两个特征融合后的向量，进而用来分类。



