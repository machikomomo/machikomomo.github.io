---
author: "momo"
date: 2022-06-08
title: "Bilinear Pooling & loss"
categories: [
    "论文笔记",
]

---

## Bilinear Pooling

双线性池化定义：

https://zhuanlan.zhihu.com/p/87650330

bilinear pooling pytorch实现：

https://github.com/HaoMood/bilinear-cnn/blob/master/src/bilinear_cnn_all.py

torch.bmm：

https://blog.csdn.net/qq_40178291/article/details/100302375（评论补充）

另一篇博客：

https://blog.csdn.net/xys430381_1/article/details/105708789（改进思路：1.降低特征维度；2.结构改变）

综上，pytorch实现可以定义如下：

```python
class BP(nn.Module):
    def __init__(self):
        super(BP, self).__init__()

    def forward(self, x1, x2):
        x1_shape = x1.size()  # 16*32*7*7
        x2_shape = x2.size()  # 16*2048*7*7
        bp = torch.einsum('imjk,injk->imn', (x1, x2))
        bp = torch.div(bp, float(x1_shape[2] * x1_shape[3]))
        bp = torch.mul(torch.sign(bp), torch.sqrt(torch.abs(bp) + 1e-12))
        bp = bp.view(x1_shape[0], -1)
        bp = F.normalize(bp, dim=-1)
        return bp
```

## Hierarchical Bilinear Pooling

跨层双线性池化：

https://blog.csdn.net/xys430381_1/article/details/105708789

（纠正，是3个特征之间两两做bilinear pooling）

vgg-16 pytorch：https://github.com/luyao777/HBP-pytorch/blob/master/HBP_all.py

resnet50-pytorch：https://github.com/Ylexx/Hierarchical-Bilinear-Pooling_Resnet_Pytorch/blob/master/hbp_model.py

备注：1.用于交互的特征是比较靠后的，eg：resnet50的feature4_0、feature4_1、feature4_2。

2.实现bilinear pooling的方式稍微有点不一样，具体可以看代码。

## Bilinear Pooling Application

https://zhuanlan.zhihu.com/p/62532887（不同场景的应用）

## Loss

**Fine-grained Recognition: Accounting for Subtle Differences between Similar Classes**

AAAI 2020 类间差异：

https://zhuanlan.zhihu.com/p/98603818?from_voters_page=true（softmax+ce----->gradient boosting loss）

softmax+ce：

https://www.zhihu.com/question/294679135

https://blog.csdn.net/zkq_1986/article/details/100668648（pytorch搭建分类网络时，不需要在fc后面（激活不激活都可）再手动添加softmax）



