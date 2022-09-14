---
author: "momo"
date: 2022-08-01
title: "clip"
categories: [
    "论文笔记",
]
---

## CLIP

github代码：https://github.com/openai/CLIP

论文地址：https://arxiv.org/abs/2103.00020

已经开放预训练好的模型，还没有开源预训练部分的代码。

相关博客：https://blog.csdn.net/weixin_44031582/article/details/122509507

https://blog.51cto.com/u_15531854/5342432

预训练部分，采用对比学习。

![截屏2022-08-04 下午2.14.21](/Users/momochan/Library/Application Support/typora-user-images/截屏2022-08-04 下午2.14.21.png)

## 用法1

输入一张图片和多个文本，得到该图片和各个文本的相似性：

```python
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50", device=device)

image = preprocess(Image.open("eye4.jpg")).unsqueeze(0).to(device)  # 1,3,224,224
text = clip.tokenize(["an eye with conjunctival congestion", "an eye with cataract",
                      "an eye with corneal dystrophy", "an eye with pterygium", "ectopia lentis",
                      "an eye with pigmented nevus"]).to(device)  # 6,77

with torch.no_grad():
    # image_features = model.encode_image(image)
    # text_features = model.encode_text(text)
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1)
    values, indices = probs[0].topk(6)
    print(values)
    print(indices)
    # Print the result
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{index}: {100 * value.item():.2f}%")

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
```



## 用法2

输入cifar10的一张图片和10个标签（转为提示板文本），得到该图片和各个文本的相似性：

```python
# import clip
# import torch
#
# print(torch.cuda.is_available())
# print(clip.available_models())
# model = clip.load('RN50', "cpu")
# print(model)

import os
import clip
import torch
from torchvision.datasets import CIFAR10

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device is " + device)
model, preprocess = clip.load('RN50', device)
print(model)
print(preprocess)

# Download the dataset
cifar10 = CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Prepare the inputs
image, class_id = cifar10[36]
print("True label is " + str(class_id))
image_input = preprocess(image).unsqueeze(0).to(device)  # torch.Size([1, 3, 224, 224])
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar10.classes]).to(device)  # torch.Size([10, 77])

with torch.no_grad():
    image_features = model.encode_image(image_input)
    print(image_features.shape)  # torch.Size([1, 1024])
    text_features = model.encode_text(text_inputs)
    print(text_features.shape)  # torch.Size([10, 1024])

# Pick the top 5 most similar labels for the image

print(text_features.norm(dim=-1, keepdim=True).shape)
print(image_features.norm(dim=-1, keepdim=True))

image_features /= image_features.norm(dim=-1, keepdim=True)
# print(image_features)
# print(image_features.shape)
text_features /= text_features.norm(dim=-1, keepdim=True)
# print(text_features)
# print(text_features.shape)
t = text_features.T
print(t)

print(t.shape)

print(100.0 * image_features @ text_features.T)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
print(similarity)
print(similarity[0])
values, indices = similarity[0].topk(5)
print(values)
print(indices)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{cifar10.classes[index]:>16s}: {100 * value.item():.2f}%")
```



## 用法3

（不使用文本/文本特征）

输入训练图片和训练标签，训练图片通过 model.encode_image() 得到训练图片的特征。

用特征和标签去训练一个逻辑回归的分类器（任意一个线性分类器）。

测试图片通过 model.encode_image() 得到测试图片的特征。输入测试图片的特征到分类器，得到预测结果。计算acc。

```python
import os
import clip
import torch

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model, preprocess = clip.load('RN50', device)
print(model)

# Load the dataset
root = os.path.expanduser("~/.cache")
train = CIFAR10(root, download=False, train=True, transform=preprocess)
test = CIFAR10(root, download=False, train=False, transform=preprocess)


def get_features(dataset):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=10)):
            print(labels)  # tensor([6, 9, 9, 4, 1, 1, 2, 7, 8, 3])

            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


# Calculate the image features
train_features, train_labels = get_features(train)
print(train_features.shape)
print(train_labels.shape)
print(train_features)
print("check out train_labels")
print(train_labels)
test_features, test_labels = get_features(test)
print(test_features.shape)
print(test_labels.shape)

# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=100, verbose=1)
classifier.fit(train_features, train_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
print(f"Accuracy = {accuracy:.3f}")
```



## 文本到文本特征

从 cataract —— an eye with cataract —— 1*77 的torch.Tensor

整个文本可以得到一个torch.Size([13, 77])的张量/矩阵，但是每个tensor有大量的空白值，这个77应该是官方定的长度。

"The context length to use; all CLIP models use 77 as the context length"

return "A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]".

```python
import clip

text_list = ["cataract",
             "intraocular lens",
             "ectopia lentis",
             "keratitis",
             "corneal scarring",
             "corneal dystrophy",
             "corneal conjunctival tumor",
             "palpebral fissure",
             "pterygium",
             "subconjunctival hemorrhage",
             "conjunctival congestion",
             "conjunctival cysts",
             "pigmented nevus"]

text_list_sentence = [f"an eye with {c}" for c in text_list]
print(text_list_sentence)
text = clip.tokenize(text_list_sentence)
print(text)
print(text.shape)
print(text[0])
```

从 torch.Size([13, 77])的张量/矩阵 —— torch.Size([13, 1024]) 的张量/矩阵/文本特征。

可能不是直接得到，而是经过一些变化再经过projector才得到1024。

一个embedding的过程。这一步可以自己选择方法去实现。因为原代码耦合性真的很强。

## 常用的embedding的方式如下

word2vec

transformer

bert

glove

elmo

 参考：https://zhuanlan.zhihu.com/p/53194407

## 得到特征以后进行训练

参考：https://blog.csdn.net/weixin_44031582/article/details/122509507

前向阶段：image输入进image_encoder得到image_features；把prompts, tokenized_prompts放进TextEncoder获得text_features，两者算相似度获得logits。

logits与label进行交叉熵运算得到loss

反向阶段：loss反向传播，优化可学习的nn.Parameter。比如prompts里的ctx以及原CLIP里的 positional_embedding。

所谓的训练得到模型，就是得到其中某些可以学习（更改权重）的参数。

对比学习（？）

## 现阶段工作——分词

1.分词（只需要实现）

https://cloud.tencent.com/developer/article/1747734

https://blog.csdn.net/sk_berry/article/details/105240317

https://zhuanlan.zhihu.com/p/390821442

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model_inputs = tokenizer(text_list_sentence, padding="longest")  # 13,15
print(model_inputs)
print(model_inputs['input_ids'])
text_input = torch.Tensor(model_inputs['input_ids'])
print(text_input)
print(text_input.shape)
```

这个分词参考下面两篇实现。

https://www.cnblogs.com/zjuhaohaoxuexi/p/15135466.html

https://blog.csdn.net/qq_28790663/article/details/115374855

## 现阶段工作——模型

预想是加动态mlp

## 相关论文

https://hub.baai.ac.cn/view/18769

## 多模态相关

https://zhuanlan.zhihu.com/p/93125122

文本信息，文件名和标签信息？



1、clip预训练模型，输入文本（标签）+图片-> 伪标签（1.准确性测定，加上topk）——考虑的是clip对文本之间的相似性？图片之间的相似性？图片+文本空间的相似性？

2、如果伪标签信息有一定的可靠性（可以有噪声），就放入多模态网络（MLP MMY），训练以及预测结果

3、直接用文件名->文本向量，嵌入到网络（文本，标签泄漏）

4、孪生网络（pairwise、triplet loss），输入image和标签文本（查找一下现有的框架、论文，是图片和文本做匹配/相似度计算）

5、clip从图像生成文本，类似视觉文本模型的应用

xclient.info



http://sujitpal.blogspot.com/2021/10/fine-tuning-openai-clip-for-different.html



https://blog.csdn.net/qq_37950540/article/details/84452642
