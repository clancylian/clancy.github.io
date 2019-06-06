---
title: 'PaperReading: FaceBoxes'
date: 2019-05-05 13:59:36
tags:
- 人脸检测
categories: 论文
top: 30
mathjax: true
---

# 概述

​	人脸检测是人脸对齐、人脸识别、人脸跟踪的基础。目前人脸检测还面临着很多挑战，主要还是精度和速度的问题。很多研究都是为了解决这两个问题。早期的方法是基于人工特征，以V-J人脸检测为代表，大量的研究都是设计一个鲁棒的特征和训练一个有效的分类器。cascade structure和DMP模型都能获得不错的性能。但是这些方法都是基于不鲁棒的特征，虽然速度快，但是在不同场景下精度不高。另一种方法是基于卷积神经网络。卷积网络对于多变人脸具有很高的鲁棒性，但是在速度上却不快，尤其是在CPU设备上面。

​	这两种方法有各自的优点。为了在速度和精度上能够达到较好的性能，一种自然的想法是结合两种类型的方法，也就是采用级联的卷积神经网络，比如MTCNN。但是基于级联的卷积神经网络会带来三个问题：1.速度和人脸数量相关，人脸越多速度越慢；2.级联检测器都是基于局部优化，训练复杂；3.不能达到实时。

​	此篇文章灵感来源于Faster R-CNN中的RPN以及SSD的多尺度机制。这是一个ONE-STAGE网络，网络结构主要包含两部分：1.Rapidly Digested Convolutional Layers (RDCL)和Multiple Scale Convolutional Layers (MSCL)。RDCL是为了解决实时问题，MSCL主要为了丰富感受野，使不同层的anchor离散化，解决人脸多尺度问题。除此之外，作者还提出Anchor densification strategy让不同类型的anchor有相同的密度，这极大提高小人脸的召回率。对于VGA图片(640x480)在CPU可以达到20FPS，在GPU可以达到125FPS。作者认为他们工作的贡献包含四部分，除了以上说的三点之外，还包含：在AFW, PASCAL face, and FDDB datasets取得最好性能(what? 这也算？)

## 网络结构	

![网络结构](https://raw.githubusercontent.com/clancylian/blogpic/master/faceboxes_framework.jpg)

### RDCL

- **缩小输入的空间大小：**为了快速减小输入的空间尺度大小，在卷积核池化上使用了一系列的大的stride，在Conv1、Pool1、Conv2、Pool2上stride分别是4、2、2、2，RDCL的stride一共是32，意味着输入的尺度大小被快速减小了32倍。
- **选择合适的kernel size：**一个网络开始的一些层的kernel size应该比较小以用来加速，同时也应该足够大用以减轻空间大小减小带来的信息损失。Conv1、Conv2以及所有的Pool层分别选取7x7，5x5，3x3的kernel size。
- **减少输出通道数：**使用C.ReLU来减少输出通道数。为啥提出这个激活函数有专门的论文参考，引：网络的前部，网络倾向于同时捕获正负相位的信息，但ReLU会抹掉负响应。 这造成了卷积核会存在冗余。

![crelu](https://raw.githubusercontent.com/clancylian/blogpic/master/crelu.jpg)

### MSCL

　　将RPN作为一个人脸检测器，不能获取很好的性能有以下两个原因：

- RPN中的anchor只和最后一个卷积层相关，其中的特征和分辨率对于处理人脸变化问题上太弱。

- anchor相应的层使用一系列不同的尺度来检测人脸，但只有单一的感受野，不能匹配不同尺度的人脸。

  

  为解决这个问题，对MSCL从以下两个角度去设计：

- **Multi-scale design along the dimension of network depth.** Anchor在多尺度的feature map上面取，类似SSD。 

- **Multi-scale design along the dimension of network width.**使用inception模块，内部使用不同大小的卷积核，可以捕获到更多的尺度信息。

![inception](https://raw.githubusercontent.com/clancylian/blogpic/master/inception.jpg)

### Anchor densification strategy

​	对于Anchor作者使用1:1的宽高比，原因是因为人脸框接近正方形。 Inception3的anchor尺度为32x32，64x64，128x128，Conv3_2、Conv4_2的尺度分别为256x256和512x512。

​	对于anchor相应层的间隔相当于步长大小、比如，对于Conv3 2步长是64，anchor大小256x256，意思是对于输入图片，每隔64个像素有一个256x256的anchor。作者提出了一个anchor密度概念：
$$
A_{density} = \frac{A_{scale}}{A_{interval}}
$$
其中分子表示anchor大小，分母表示anchor间隔，对于anchor间隔一般是默认的，也就是步长大小，分别为32、32、32、64、128。根据式子计算出来的密度分别为1、2、4、4、4。由此可以看到对于小人脸anchor太稀疏，密度太低，会导致小人脸的召回率下降。为了消除这个不平衡，作者提出了一种策略，在原来的anchor中心均匀叠加n^2个anchor，以保证密度相同，所以对于32x32的anchor叠加为原来的４倍，对于64x64的anchor叠加为原来的２倍。

![anchor](https://raw.githubusercontent.com/clancylian/blogpic/master/anchor-expand.jpg)

## 训练

### 训练集

WIDER FACE的子集，12880个图片。

### 数据增强

- Color distortion
- Random cropping
- Scale transformation
- Horizontal flipping
- Face-box filter

### 匹配策略

在训练时需要判断哪个anchor是和哪个bounding box对应。首先使用jaccard overlap将每个脸和anchor对应起来，然后对anchor和任意脸jaccard overlap高于阈值（0.35）的匹配起来。

### 损失函数

和Faster R-CNN中的RPN用同样的loss，一个2分类的softmax loss用来做分类，smooth L1用来做回归。

### **Hard negative mining:**

在anchor匹配后，大多数anchor都是负样本，导致正样本和负样本严重不均衡。为了更快更稳定的训练，将他们按照loss值排序并选取最高的几个，保证正样本和负样本的比例最高不超过3:1.

### **Other implementation details:**

Xavier随机初始化。优化器SGD，momentum:0.9，weight decay:5e-4，batch  size:32，迭代最大次数:120k，初始80k迭代learning  rate:1e-3，80-100k迭代用1e-4，,100-120k迭代用1e-5，使用caffe实现。



## 参考链接

[《Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units》](http://cn.arxiv.org/abs/1603.05201)

[CReLU激活函数](https://blog.csdn.net/shuzfan/article/details/77807550)

[代码](https://github.com/sfzhang15/FaceBoxes)