---
title: PyTorch学习之路
date: 2019-04-13 17:33:48
tags:
- Pytorch
- 深度学习框架
categories: 深度学习
top: 12
---

[参考文档](https://pytorch.org/tutorials/)

## 1. 基础概念

PyTorch类似于Numpy。它的优点在于可以充分利用GPU资源。它是一个深度学习框架，提供最大的灵活性和速度。

### 1.1 张量概念

```python
from __future__ import print_function
import torch

# 构造未初始化5x3的矩阵
x = torch.empty(5, 3)
print(x)

# 构造随机初始化矩阵
x = torch.rand(5, 3)
print(x)

# 构造0矩阵，数据类型为long
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# 直接用数据构造一个张量
x = torch.tensor([5.5, 3])
print(x)

# 基于存在的张量构造，可以使用存在的张量的一些属性，比如数据类型，数据纬度，除非手动修改
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size

# 获取张量大小.实际上torch.Size是一个元组类型，支持元组所有操作
print(x.size())
```

### 1.2 张量操作

关于张量的操作有100多种，包括转置、索引、切片、数值计算、线性代数、随机数等等。

```python
# 加法1
y = torch.rand(5, 3)
print(x + y)
# 加法2
print(torch.add(x, y))

# 输出张量作为参数
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# 原地（in-place）加法
# adds x to y
y.add_(x)
print(y)
# 以 _ 作为后缀将会改变操作数，比如x.copy_(y), x.t_(),都会改变x的值

# 可以像Numpy一样索引
print(x[:, 1])

# resize和reshape操作可以使用torch.view
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

# 如果你有一个只有一个元素的张量，可以使用.item()获取数值
x = torch.randn(1)
print(x)
print(x.item())
```

### 1.3 与Numpy互操作

```python
# tensor转为Numpy
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

# Numpy转为tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
```

### 1.4 CUDA 张量

可以使用 .to 方法把张量拷贝到设备内存(GPU)

```python
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
device = torch.device("cuda")          # a CUDA device object
y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
x = x.to(device)                       # or just use strings ``.to("cuda")``
z = x + y
print(z)
print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
```

### 1.5 自动微分技术

PyTorch中的反向传播中求导都是使用autograd包完成的。它提供了张量求导所有操作，它是一个define-by-run框架，意味着每一次反向传播都取决于你的代码是如何跑的，每一次迭代都可能不同。

`torch.Tensor` 是pytorch一个最基础的类。如果你设置其属性 `.requires_grad` 为 `True`, 它将会跟踪它所有的操作，当你调用`.backward()`时，会自动计算梯度，可以使用 `.grad` 获取梯度值。可以使用`.detach()`来停止跟踪历史或者阻止将来的操作。也可以使用`with torch.no_grad():`。

除了`torch.Tensor` 还有一个很重要的类来实现自动求导——`Fucction`。每一个张量（除了用户创建的之外）都有`.grad_fn`属性。

当想要计算导数的时候只要调用`.backward()`。当输出张量是一个标量的时候，不需要特别指明参数，然而如果是一个向量就需要指明`gradient`参数。

```python
# 设置requires_grad为true
x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)
# 因为y是由操作得来的结果，所以有grad_fn属性
print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z, out)

# 因为out输出为标量，相当于out.backward(torch.tensor(1.))
out.backward()
print(x.grad)
```

### 1.6 神经网络

#### 1.6.1 定义网络结构

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

def __init__(self):
super(Net, self).__init__()
# 1 input image channel, 6 output channels, 5x5 square convolution
# kernel
self.conv1 = nn.Conv2d(1, 6, 5)
self.conv2 = nn.Conv2d(6, 16, 5)
# an affine operation: y = Wx + b
self.fc1 = nn.Linear(16 * 5 * 5, 120)
self.fc2 = nn.Linear(120, 84)
self.fc3 = nn.Linear(84, 10)

def forward(self, x):
# Max pooling over a (2, 2) window
x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
# If the size is a square you can only specify a single number
x = F.max_pool2d(F.relu(self.conv2(x)), 2)
x = x.view(-1, self.num_flat_features(x))
x = F.relu(self.fc1(x))
x = F.relu(self.fc2(x))
x = self.fc3(x)
return x

def num_flat_features(self, x):
size = x.size()[1:]  # all dimensions except the batch dimension
num_features = 1
for s in size:
num_features *= s
return num_features


net = Net()
# 输出网络结构
print(net)

# 模型参数存在net.parameters()中
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

# 测试输入
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# 模型梯度置0然后反向传播
net.zero_grad()
out.backward(torch.randn(1, 10))
```

**注：torch.nn只支持最小批量**

#### 1.6.2 损失函数

```python
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
```

由此可以得出计算图：

```python
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
-> view -> linear -> relu -> linear -> relu -> linear
-> MSELoss
-> loss
```

#### 1.6.3 反向传播

```python
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
```

#### 1.6.4 更新权重

```python
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
```

**注：每次`optimizer.zero_grad()`需要手动置0，因为梯度是累积的。**

### 1.7 训练一个分类器

#### 1.7.1 加载数据

```python
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
[transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

#### 1.7.2 定义卷积神经网络

```python
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
def __init__(self):
super(Net, self).__init__()
self.conv1 = nn.Conv2d(3, 6, 5)
self.pool = nn.MaxPool2d(2, 2)
self.conv2 = nn.Conv2d(6, 16, 5)
self.fc1 = nn.Linear(16 * 5 * 5, 120)
self.fc2 = nn.Linear(120, 84)
self.fc3 = nn.Linear(84, 10)

def forward(self, x):
x = self.pool(F.relu(self.conv1(x)))
x = self.pool(F.relu(self.conv2(x)))
x = x.view(-1, 16 * 5 * 5)
x = F.relu(self.fc1(x))
x = F.relu(self.fc2(x))
x = self.fc3(x)
return x


net = Net()
```

#### 1.7.3 定义损失函数和优化器

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

#### 1.7.4 训练网络

```python
for epoch in range(2):  # loop over the dataset multiple times

running_loss = 0.0
for i, data in enumerate(trainloader, 0):
# get the inputs
inputs, labels = data

# zero the parameter gradients
optimizer.zero_grad()

# forward + backward + optimize
outputs = net(inputs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()

# print statistics
running_loss += loss.item()
if i % 2000 == 1999:    # print every 2000 mini-batches
print('[%d, %5d] loss: %.3f' %
(epoch + 1, i + 1, running_loss / 2000))
running_loss = 0.0

print('Finished Training')
```

#### 1.7.5 测试

```python
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
for data in testloader:
images, labels = data
outputs = net(images)
_, predicted = torch.max(outputs, 1)
c = (predicted == labels).squeeze()
for i in range(4):
label = labels[i]
class_correct[label] += c[i].item()
class_total[label] += 1


for i in range(10):
print('Accuracy of %5s : %2d %%' % (
classes[i], 100 * class_correct[i] / class_total[i]))
```

```
Accuracy of plane : 72 %
Accuracy of   car : 47 %
Accuracy of  bird : 41 %
Accuracy of   cat : 32 %
Accuracy of  deer : 42 %
Accuracy of   dog : 49 %
Accuracy of  frog : 70 %
Accuracy of horse : 62 %
Accuracy of  ship : 46 %
Accuracy of truck : 76 %
```

#### 1.7.6 GPU训练

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

net.to(device)
inputs, labels = inputs.to(device), labels.to(device)
```

### 1.8 数据并行

```python
# cuda设备
device = torch.device("cuda:0")

# 将模型移到GPU
model.to(device)

# 输入拷贝到GPU
mytensor = my_tensor.to(device)

# 设置数据并行
model = nn.DataParallel(model)
```



------



## 2. 数据

## 3. 模型

## 4. 策略（损失函数）

## 5. 算法





