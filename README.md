# “华为云杯”2020人工智能创新应用大赛-季军方案

本赛题任务是基于高分辨可见光遥感卫星影像，提取复杂场景的道路与街道网络信息，将影像的逐个像素进行前、背景分割，检测所有道路像素的对应区域。
[大赛官网](https://competition.huaweicloud.com/information/1000041322/circumstance)

初赛成绩：0.8377 &nbsp; (6/377)

## 环境
* Ubuntu 18.04 1080Ti
* Python 3.7
* Pytorch 1.4
* albumentations

## 数据集
使用baseline默认的切图代码。切图的边长为512，步长为256，训练验证集比例为10:1。

之后通过筛再去除全黑的图像作为最终用于训练和验证的数据集。

## 网络结构
EfficientUnet-b3 + EfficientUnet-b4

[模型代码链接](https://github.com/zhoudaxia233/EfficientUnet-PyTorch)

## 涨点技巧
* 数据增强
* loss权重
* 模型融合
* 测试时增强（TTA）
* 忽略边缘预测 &nbsp; [知乎链接](https://zhuanlan.zhihu.com/p/158769096)

## 具体流程
### 1. 生成数据集
切分图片：打开others/cut_data.py，修改数据集地址data_dir，运行生成数据集。注意，这里生成的图片是BGR通道的。

去除黑边：打开others/remove_black.py, 填写刚刚生成的数据集地址old_data_dir，选择去除黑边后数据集新保存的地址data_dir。

### 2. 训练
模型训练：打开train.py，修改数据集地址data_dir，选择训练的模型（默认是b4），运行。生成的权重以及日志保存在outputs文件里。

### 3. 部署
模型融合：打开others/model_fusion.py，更改b3、b4权重文件路径，运行生成集成模型的权重文件。

上传部署：将该.pth模型文件放到submission文件夹中，然后上传至modelarts部署。

