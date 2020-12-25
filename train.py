import os
import torch
import numpy as np

from model import *
from PIL import Image
from utile import train_net
from dataset import RSCDataset
from dataset import train_transform, val_transform

Image.MAX_IMAGE_PIXELS = 1000000000000000

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 准备数据集
data_dir = "/media/inch/ubuntu/data/Competition/data/RSC_data"
train_imgs_dir = os.path.join(data_dir, "train/images/")
val_imgs_dir = os.path.join(data_dir, "val/images/")

train_labels_dir = os.path.join(data_dir, "train/labels/")
val_labels_dir = os.path.join(data_dir, "val/labels/")

train_data = RSCDataset(train_imgs_dir, train_labels_dir, transform=train_transform)
valid_data = RSCDataset(val_imgs_dir, val_labels_dir, transform=val_transform)

# 网络

#model_name = 'efficient-b3'
#model = get_efficientunet_b3(out_channels=2).to(device)

model_name = 'efficient-b4'
model = get_efficientunet_b4(out_channels=2).to(device)

# 模型保存路径
save_ckpt_dir = os.path.join('./outputs/', model_name, 'ckpt')
save_log_dir = os.path.join('./outputs/', model_name)
if not os.path.exists(save_ckpt_dir):
    os.makedirs(save_ckpt_dir)
if not os.path.exists(save_log_dir):
    os.makedirs(save_log_dir)

# 参数设置
param = {}

param['epochs'] = 80          # 训练轮数
param['batch_size'] = 4       # 批大小
param['lr'] = 1e-3            # 学习率
param['gamma'] = 0.2          # 学习率衰减系数
param['step_size'] = 5        # 学习率衰减间隔
param['momentum'] = 0.9       # 动量
param['weight_decay'] = 0.    # 权重衰减
param['disp_inter'] = 1       # 显示间隔(epoch)
param['save_inter'] = 1       # 保存间隔(epoch)
param['iter_inter'] = 500     # 显示迭代间隔(batch)

param['model_name'] = model_name          # 模型名称
param['save_log_dir'] = save_log_dir      # 日志保存路径
param['save_ckpt_dir'] = save_ckpt_dir    # 权重保存路径

# 加载权重路径（继续训练）
param['load_ckpt_dir'] = None


# 训练
best_model, model = train_net(param, model, train_data, valid_data)

