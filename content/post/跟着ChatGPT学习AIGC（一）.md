---
author: "momo"
date: 2023-03-18
title: "跟着ChatGPT学习AIGC（一）"
categories: [
    "AIGC",
    "ChatGPT"
]

---

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-186.06.07.png" style="zoom:50%;" />

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-186.08.56.png)

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-186.08.56.png)

![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-186.13.05.png)



## ArtBreeder

https://www.artbreeder.com/

1.官网的界面/UI设计非常有特色啊……

2.可操作性：非常简单，5-10min可以上手，唯一的难点在于有看不懂的英文，扔进deepl进行翻译。

- 左侧工具栏：选择、画笔（可以画一些形状）、裁剪（就是剪刀的使用）、图片（支持本地上传/拖拽进面板）、回退和前进操作。
- 中间下方的面板：输入prompt（提示词）以后可以进行渲染。支持调整参数和多组prompt。
- 主界面：对图片去进行一些操作，比如裁剪、拖拽等。
- 右侧工具栏：快速选定元素。调整面板的比例（放大、缩小）。

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-186.36.03.png" style="zoom:50%;" />

3.实现效果：拉啊、太拉了……（一定是我使用的方式不对……）这是我在它的示范样例基础上略加修改的prompt。

本意：1.生成一个有猫耳的调皮的机器人。2.这个机器人在秋天感到失落（我觉得这句prompt写得就不对劲，不是很具象的一句提示）。3.参数随便填的。

它生成的效果一览（支持多次生成）：

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-186.44.49.png" style="zoom:50%;" />

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-186.45.13.png" style="zoom:50%;" />

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-186.47.01.png" style="zoom:50%;" />

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-186.47.57.png" style="zoom:50%;" />



4.啊啊啊不懂啊，我的猫耳呢，我的猫耳元素怎么被忽略掉了。想到第二句promt可能写得不好，所以决定删除。目标是生成一个有猫耳的机器人。

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-186.52.24.png" style="zoom:50%;" />

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-186.53.45.png" style="zoom:50%;" />

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-186.54.25.png" style="zoom:50%;" />

5.好像有点像样了，但是和我想的差太远了……来调整一下参数。看看参数不同的影响。

控制其他参数不变，调整 AI render intensity（这个没有具体的参数值，就拖着进度条大概取了0、1/4、1/2、3/4、1）

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-186.57.01.png" style="zoom:50%;" />

6.下面是AI render intensity分别取0、1/4、1/2对应的结果。

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-187.00.49.png" style="zoom:50%;" />

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-187.02.50.png" style="zoom:50%;" />

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-187.03.17.png" style="zoom:50%;" />

7.AI render intensity取到3/4的时候，有点符合（我）心里的预期了。总体上就是卡通风格。于是多取了几次，reroll的过程其实就是去修改另一个参数（seed）的过程。所以这个reroll其实是一种假随机吧，通过修改seed来生成不同的图。但是只要参数不变的话，生成的图像是不会发生改变的。

以下是AI render intensity取3/4不变，reroll增大分别得到的图片。感觉也没有什么规律，直白地讲可能是不同的画风？

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-187.04.17.png" style="zoom:50%;" />

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-187.05.11.png" style="zoom:50%;" />

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-187.04.56.png" style="zoom:50%;" />

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-187.12.11.png" style="zoom:50%;" />

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-187.14.07.png" style="zoom:50%;" />

8.最后是AI render intensity取到1，seed取10、20、40、60、80、99得到的图像，嗯，不乏一些比较抽象的……

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-187.15.42.png" style="zoom:50%;" />

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-187.16.28.png" style="zoom:50%;" />

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-187.17.12.png" style="zoom:50%;" />

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-187.18.14.png" style="zoom:50%;" />

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-187.18.50.png" style="zoom:50%;" />

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-187.19.12.png" style="zoom:50%;" />

9.总的来讲，AI render intensity越高，生成的图像的自由度越高，应该是模型会使用自己的素材库，去替换掉原面板里的图像。

seed可以去生成不同的风格。但是话说回来，如果AI render intensity取值不高也就是说原面板里的图像保留较多的时候，seed可以发挥的空间是不多的。如下，AI render intensity取值0，调整seed为20、40、60、80，并没有什么明显差别，因为风格无法在既定的素材里得到展示。



<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-187.23.09.png" style="zoom:50%;" />

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-187.24.08.png" style="zoom:50%;" />

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-187.24.34.png" style="zoom:50%;" />

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-187.25.38.png" style="zoom:50%;" />

但是在AI render intensity取到3/4时，调整seed从20、40、60、80，风格的差异就能体现。话说为啥我觉得seed是风格呢，因为有时候甚至能在图片右下角看到一些可能是原有创作者的签名……

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-187.26.22.png" style="zoom:50%;" />

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-187.27.18.png" style="zoom:50%;" />

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-187.27.37.png" style="zoom:50%;" />

<img src="https://halfbit.oss-cn-hangzhou.aliyuncs.com/2023-03-187.28.15.png" style="zoom:50%;" />

10.总体上，上手不是很难呢，但是要去调整参数、寻找素材，最后得到一个符合自己心理预期的图片，还是需要花时间的呢。所以如果有人能提供一些训练的技巧（可以这么说吧），比如说某种风格的参数是多少左右比较合适，那应该会减少这个训练的成本。虽然也可以很粗暴地比如说提供一些标签词（xx风）去限定生成的图片的风格，那样应该更方便，不过目前这种方式自由度应该也更高一点吧！

11.下一步应该是找一些相关的教程，怎样去更好地用它训练出指定风格的图片，调整参数的技巧是什么。
