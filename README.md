# “华为云杯”2020人工智能创新应用大赛-季军方案

本赛题任务是基于高分辨可见光遥感卫星影像，提取复杂场景的道路与街道网络信息，将影像的逐个像素进行前、背景分割，检测所有道路像素的对应区域。
[大赛官网](https://competition.huaweicloud.com/information/1000041322/circumstance)

初赛成绩：0.8377 &nbsp; (6/377)

## 环境
* Ubuntu 18.04 
* Python 3.7
* Pytorch 1.4
* albumentations

## 数据集
使用baseline默认的切图代码。切图的边长为512，步长为256，训练验证集比例为10:1。

之后通过筛再去除全黑的图像作为最终用于训练和验证的数据集。

## 涨点技巧
* 数据增强
* 模型融合
* TTA
* 忽略边缘预测 &nbsp; [知乎链接](https://zhuanlan.zhihu.com/p/158769096)


