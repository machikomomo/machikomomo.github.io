---
author: "momo"
date: 2022-09-07
title: "Transformer相关"
categories: [
    "论文笔记",
]
---

## nn.Transformer

最简单的实现，利用pytorch封装好的类。

参考：https://zhuanlan.zhihu.com/p/107586681

```python
import torch.nn as nn
import torch

transformer = nn.Transformer(d_model=512, nhead=8)
src = torch.rand((49, 12, 512))
tgt = torch.rand((14, 12, 512))
out = transformer(src, tgt)
print(out.shape)
```

src是编码器的输入，tgt是解码器的输入。out的形状和tgt的形状一致。



## query2label里面是如何使用transformer结构的

1、stage1：获取backbone提取出来的特征比如，（12，2048，7，7）-> （49，12，256）

2、stage2：构建一个参数可更新的label_embedding，（14，12，256）

3、transformer（feature_maps，label_emb），形状为（14，12，256）-> （12，14，256）-> （12，14*256）

4、最后一步，经过全连接层，得到（12，num_classes）的tensor，最后是要过softmax还是sigmoid都可以，选择不同的loss。



我想要知道的，其实是对于feature_maps和label_emb这两个tensor，中间经过了怎样的黑盒子，以及论文的意思是否对得上。

所以，需要了解nn.Transformer中编码器的结构和解码器的结构。在原文中，编码器个数为1，解码器个数为2。



## 尝试参考nn.Transformer源码

了解大意即可。按照2017年的Attention is all you need所编写的标准的代码。



## EncoderLayer

由自注意力（由多头注意力机制实现）和前馈神经网络所构成。

```python
import torch
import torch.nn as nn

encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
src = torch.rand((49, 12, 256))
out = encoder_layer(src)
print(out.shape)

'''
输出结果为 torch.Size([49, 12, 256]) 说明输入和输出的形状是完全一致的
'''
```

黑盒子，里面对这个src做了哪些处理？注意，虽然这里可以穿入三个参数，src、src_mask、src_key_padding_mask但是实际上用到的只有src，只传入了src，所以代码其实可以简化。

```python
def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
    r"""Pass the input through the encoder layer.

    Args:
        src: the sequence to the encoder layer (required).
        src_mask: the mask for the src sequence (optional).
        src_key_padding_mask: the mask for the src keys per batch (optional).

    Shape:
        see the docs in Transformer class.
    """
    src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                          key_padding_mask=src_key_padding_mask)[0]
    '''
    src2 = self.self_attn(src,src,src) 相当于对这个自注意力层，传入了三个一摸一样的tensor
    '''
    src = src + self.dropout1(src2) # 这里相当于做一个残差连接吧，src+正则化以后的src2
    src = self.norm1(src) # 这里的norm都是LN
    src2 = self.linear2(self.dropout(self.activation(self.linear1(src)))) # 线性层、relu激活函数、dropout、线性层
    src = src + self.dropout2(src2) # 依然是一个残差连接
    src = self.norm2(src) # LN
    return src
```



## MultiheadAttention

transformer本质上最重要的应该就是这一块。多头注意力机制。

```python
multihead_attn = nn.MultiheadAttention(256, 8)
attn_output, attn_output_weights = multihead_attn(src,src,src)
print(src.shape)
print(attn_output.shape)
print(attn_output_weights.shape)
'''
torch.Size([49, 12, 256])
torch.Size([49, 12, 256])
torch.Size([12, 49, 49])
'''
```

输入是上面那个src，连着3个一样的。输出，第一个是真正需要的输出，第二个应该就是权重、重要程度、关注度啥的，anyway，

输入和输出的形状是一样的。三个输入分别是q k v，里面的操作，说实话不感兴趣了，因为不会改里面的，所以。在编码层，就是三个一样的src传了进去，对吧？



## DecoderLayer

由自注意力（多头注意力机制实现）和前馈神经网络、互注意力机制（也是多头注意力机制实现）所构成。

```
decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=8)
memory = torch.rand(10, 32, 256)
tgt = torch.rand(20, 32, 256)
out = decoder_layer(tgt, memory)
print(out.shape)
```

注意一下，这个层的输入顺序是，先tgt，再是（解码器的输出）。

输出的大小和tgt的大小是一致的。

黑盒子。

```python
def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
    r"""Pass the inputs (and mask) through the decoder layer.

    Args:
        tgt: the sequence to the decoder layer (required).
        memory: the sequence from the last layer of the encoder (required).
        tgt_mask: the mask for the tgt sequence (optional).
        memory_mask: the mask for the memory sequence (optional).
        tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
        memory_key_padding_mask: the mask for the memory keys per batch (optional).

    Shape:
        see the docs in Transformer class.
    """
    tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                          key_padding_mask=tgt_key_padding_mask)[0]
    tgt = tgt + self.dropout1(tgt2)
    tgt = self.norm1(tgt)
    tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
    tgt = tgt + self.dropout2(tgt2)
    tgt = self.norm2(tgt)
    tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    tgt = tgt + self.dropout3(tgt2)
    tgt = self.norm3(tgt)
    return tgt
```

先是自注意力机制（扔进去tgt、tgt、tgt）实现一个tgt2，然后dropout正则化一下再残差连接；然后是LN；

然后是互注意力机制（扔进去tgt作为query，编码器的输出作为key和value）生成tgt2。

然后tgt=tgt+dropout（tgt2）

然后LN、线性层、激活、dropout、线性层，再残差连接、norm。

可以理解，其实只是说，核心区别，就是q、k、v里面的k和v换成了之前编码层的输出。



## 总结

单独的编码器：对src*3进行自注意力操作。

单独的解码器：对tgt*3进行自注意力操作，然后对tgt、memory、memory进行互注意力操作。

编码器的堆叠：重复一样的操作。

解码器的堆叠：经过一个解码器，相当于更新了tgt；然后下一个解码器的时候，放入原来的memory和更新后的tgt。

transformer：需要定义几个编码器，几个解码器，然后对src进行编码器的操作，结束以后，这个src就作为memory，输入解码器。

每一个解码器的memory都是一样的，即key、value是不变的，变的是解码器的输入即tgt即query。它会更新。



## 最后的总结

看源码是最好的，最简洁的。

mask和pad在cv领域的transformer里面（query2label里面）是没有用到的，不用考虑。

最后再看一下query2label的插图。

可以看到query也就是tgt是会更新的。但是key&value始终是编码器（图中省略了编码器）的输出。



![](https://halfbit.oss-cn-hangzhou.aliyuncs.com/2022-09-141.49.52.png)
