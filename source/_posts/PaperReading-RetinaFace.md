---
title: 'PaperReading:RetinaFace'
date: 2019-05-07 14:17:11
tags:
- 人脸检测
categories: 论文
top: 30
---

## 摘要

虽然对于无约束人脸检测取得了巨大进步，但是精度和速度依然是一个挑战。作者提出一种鲁棒的single-stage人脸检测器，加入extra-supervised和self-supervised模块，提高人脸检测性能的多任务学习方法。论文主要贡献有５点：

- 手动标注WIDER FACE数据集人脸关键点，并且在难人脸检测上由于额外监督信号的帮助取得了巨大的提高。
- 在现有的模块上并行加入self-supervised解码分支预测3D人脸信息分支。
- 在WIDER FACE hard test数据集上提高1.1%（达到91.4%）。
- 在IJB-C test set上测试结果表明可以提高ArcFace在人脸验证精度(TAR=89.59% for FAR=1e-6)。
- 使用轻量级骨干网络，RetinaFace在CPU上测试VGA图片可以达到实时。

## 概述

此片论文包含face detection、face alignment、pixel-wise face parsing、3D dense correspondence regres-sion等任务。

首先论文灵感来源于一般目标检测rcnn系列、SSD、YOLO系列、FPN、Focal loss。和一般的目标检测不同，人脸检测的宽高比一般是1:1到1:1.5之间，但是大小可以从几个像素到几千个像素。目前比较流行的方法是one-stage方法，速度比较快，所以作者基于one-stage采用多任务方法得到state-of-the-art结果。

《Joint cascade face detection and alignment》论文提出的联合人脸检测和人脸对齐可以提取到更好的人脸特征。所以基于MTCNN和STN方法灵感，作者加入５个人脸关键点，由于训练数据的限制JDA 、MTCNN、和STN没有验证过小人脸检测是否可以从额外的５个关键点中获益。通过加入５个关键点，作者想看看能否在WIDER FACE hard test取得更好的性能。

Mask R-CNN通过加入了Mask并行预测分支之后性能得到了很大的提升。证实了密集pixel-wise标签可以提高检测。然而WIDER FACE对密集标注很难实施，能够使用非监督方法来进一步提高人脸检测呢？

在FAN论文中，提出了一种anchor-level attention map来提高遮挡的人脸检测，但是这个方法太粗糙，并且不包含语义信息。目前，self-supervised 3D morphable models在3D人脸建模取得良好成绩，尤其是Mesh Decoder达到了实时。但是应用Mesh Decoder方法有两个难点：１相机参数难以估计精确；２特征漂移。本篇论文作者使用自监督学习方法加入额外分支来预测3D人脸形状。

## 相关工作

**Image pyramid v.s. feature pyramid**

**Two-stage v.s. single-stage**

**Context Modelling**

**Multi-task Learning**



## RetinaFace

### 多任务损失函数

$$
L = L_{cls}(p_i ,p^∗_i) + λ_1p^∗_iL_{box}(t_i,t^∗_i)
+ λ_2p^∗_iL_{pts}(l_i,l_i^∗) + λ_3p^∗_iL_{pixel}
$$

其中Lcls代表分类损失，softmax，pi表示anchor i预测为人脸的概率，p*为１表示为正样本，为０表示负样本。

其中Lbox为边框回归损失，smooth-L1，t i = {t x , t y , t w , t h }，对中心点和宽高进行归一化操作，对于正样本anchor的损失。

其中L pts为５个关键点回归，类似边框回归中的中心点回归。

其中L pixel为密集回归损失函数，具体函数见后面。

λ 1 -λ 3为权重分别设置为0.25 0.1 0.01