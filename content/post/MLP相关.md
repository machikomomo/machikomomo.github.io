---
author: "momo"
date: 2022-09-03
title: "MLP相关"
categories: [
    "论文笔记",
]
---

## MLP

博客：https://zhuanlan.zhihu.com/p/63184325

pytorch：https://blog.csdn.net/rocketeerLi/article/details/92158767 改正如下

https://blog.csdn.net/geter_CS/article/details/84857220

mnist：https://zhuanlan.zhihu.com/p/36592188



## code

```python
'''
mlp 最后结果
epoch:19	avg training loss:0.022919857291857866
accuracy of val dataset is 97.88
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

# 定义全局变量
n_epochs = 20  # epoch 的数目
batch_size = 16  # 决定每次读取多少图片

# 定义训练集 测试集，如果找不到数据，就下载
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

print('data loaded!')

# 创建加载器
train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=0)
test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=0)


# 建立一个四层感知机网络
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # 2个隐藏层 1个输出层
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # ---> (bs,784)
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        # out = F.softmax(out, dim=1) 这个是不需要的
        return out


def train():
    # 定义损失函数和优化器
    model = MLP()
    print('model loaded!')
    loss_fc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

    for epoch in range(n_epochs):
        # train
        model.train()
        train_loss = 0.0
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fc(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        train_loss = train_loss / (len(train_loader.dataset))
        print(f'epoch:{epoch}\tavg training loss:{train_loss}')

        # val
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                output = F.softmax(output, dim=1)
                _, predicted = torch.max(output.data, 1)
                total += data.size(0)
                correct += (predicted == target).sum().item()
        print(f'accuracy of val dataset is {100.0 * correct / total}')

if __name__ == '__main__':
    train()
```



## 动态mlp

博客：https://blog.csdn.net/odssodssey/article/details/124099498

(bs,channels) 大小的img_feature 和 text_feature 融合
