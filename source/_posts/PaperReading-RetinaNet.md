---
title: 'PaperReading:RetinaNet'
date: 2019-05-10 14:07:25
tags:
- 目标检测
categories: 论文
top: 30
mathjax: true
---

# Focal Loss for Dense Object Detection

[代码链接](https://github.com/facebookresearch/Detectron)　[论文链接](https://arxiv.org/abs/1708.02002)

作者提出了focal loss，通过在交叉熵损失函数中加入一个衰减系数，达到可以降低那些”分类好“的样本的损失，从而更加关注那个误分类的样本。根据这个损失函数，作者训练了一个检测器RetinaNet，性能比当前最好的two-stage检测器性能还高，在COCO数据集可以达到40.8 AP。

之前最好的目标检测器还是基于two-stage，基于候选框驱动机制。先生成候选框目标集，然后在分类回归定位。那么one-stage是否也可以达到同样的精度呢？one-stage通过密集采样，基于anchor机制采样不同目标位置，不同scale，不同ratios。比如YOLO，SSD等方法。

作者提出的RetinaNet可以匹敌two-stage检测器，比如Feature Pyramid Network (FPN)、Mask R-CNN、Faster R-CNN。作者发现了类别不均衡是导致one-stage检测器精度上不去的原因。在two-stage检测器中一般有提取候选框的步骤，所以可以很快降低候选框数目，这其实以及有降低类别不均衡的意思在里面，因为图像大部分地方都是背景，在two-stage方法中解决样本不均衡一般采用启发式采样，比如前景和背景比例保持在1:3左右，或者使用难样本挖掘（online hard example mining (OHEM)）。然而在one-stage中一般要处理100k左右的候选框，虽然也使用启发式采样，但是效率很低，比如使用bootstrapping或者hard example mining等方法。

样本不均衡所带来的问题：1训练效率不高，因为大部分的样本为easy-negatives，对学习没有帮助；2大量的easy-negatives会使训练overwhelm从而使模型退化。

很多论文都在解决损失函数鲁棒性问题，比如Huber loss，但是大部分的损失函数都是致力于离群点，降低hard examples的损失权重，相反的focal loss主要设计为处理样本不均衡问题，致力于降低非离群点也就是easy example的损失权重。

## Focal Loss

标准交叉熵损失函数：
$$
CE(p,y)= \left\{
\begin{aligned}
-log(p) && if(y = 1) \\
-log(1-p) && otherwise 
\end{aligned}
\right.
$$
设
$$
p_t =
\left\{
\begin
{aligned}
p && if(y = 1) \\
1-p && otherwise
\end
{aligned}
\right.
$$
可以得到CE(p, y) = CE(pt ) = − log(pt )

画出曲线可以分析看到即使是容易分类的样本，数量一多，也会造成很大的损失。实验表明样本不均衡使得易分类样本的损失占大部分，主导了梯度。作者提出的损失函数：
$$
FL(p_t ) = −(1 − p_t )^γ log(p_t )
$$


当样本误分类的时候pt很小，系数接近于１，系数对loss没啥影响；当pt接近于１的时候，系数接近０，对于易分类的样本可以降低权重。通过降低易分类样本的损失权重反过来增加了分类错误的损失的重要性。

作者还加了一个系数可以提高精度：
$$
FL(p_t ) = −α_t (1 − p_t )^γ log(p_t )
$$
![图片](https://raw.githubusercontent.com/clancylian/blogpic/master/retinanet.png)