import os
import time
import copy
import torch
import random
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim

from glob import glob
from PIL import Image
from tqdm import tqdm

from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from utile.utils import AverageMeter, second2time, inial_logger
from albumentations.augmentations import functional as F


Image.MAX_IMAGE_PIXELS = 1000000000000000


def smooth(v, w=0.85):
    last = v[0]
    smoothed = []
    for point in v:
        smoothed_val = last * w + (1 - w) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def train_net(param, model, train_data, valid_data, plot=False):
    # 初始化参数
    model_name      = param['model_name']
    epochs          = param['epochs']
    batch_size      = param['batch_size']
    lr              = param['lr']
    gamma           = param['gamma']
    step_size       = param['step_size']
    momentum        = param['momentum']
    weight_decay    = param['weight_decay']

    disp_inter      = param['disp_inter']
    save_inter      = param['save_inter']
    iter_inter      = param['iter_inter']

    save_log_dir    = param['save_log_dir']
    save_ckpt_dir   = param['save_ckpt_dir']
    load_ckpt_dir   = param['load_ckpt_dir']

    # 网络参数
    train_data_size = train_data.__len__()
    valid_data_size = valid_data.__len__()
    c, y, x = train_data.__getitem__(0)['image'].shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=4)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 55, 60, 70, 80], gamma=gamma)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([3, 1]), reduction='mean').to(device)
    #criterion = nn.BCEWithLogitsLoss()
    logger = inial_logger(os.path.join(save_log_dir, time.strftime("%m-%d %H:%M:%S", time.localtime()) +'_'+model_name+ '.log'))

    # 主循环
    train_loss_total_epochs, valid_loss_total_epochs, epoch_lr = [], [], []
    train_loader_size = train_loader.__len__()
    valid_loader_size = valid_loader.__len__()
    best_loss = 1e50
    best_mode = copy.deepcopy(model)
    epoch_start = 0
    if load_ckpt_dir is not None:
        ckpt = torch.load(load_ckpt_dir)
        epoch_start = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

    logger.info('Total Epoch:{} Image_size:({}, {}) Training num:{}  Validation num:{}'.format(epochs, x, y, train_data_size, valid_data_size))

    for epoch in range(epoch_start, epochs):
        epoch_start = time.time()
        # 训练阶段
        model.train()
        train_epoch_loss = AverageMeter()
        train_iter_loss = AverageMeter()
        for batch_idx, batch_samples in enumerate(train_loader):
            data, target = batch_samples['image'], batch_samples['label']
            data, target = Variable(data.to(device)), Variable(target.to(device))
            optimizer.zero_grad()
            pred = model(data)
            # print(pred.shape, target.shape)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            image_loss = loss.item()
            train_epoch_loss.update(image_loss)
            train_iter_loss.update(image_loss)
            if batch_idx % iter_inter == 0:
                logger.info('[train] epoch:{} iter:{}/{} {:.2f}% loss:{:.6f}'.format(
                    epoch, batch_idx, train_loader_size, batch_idx/train_loader_size*100, train_iter_loss.avg))
                train_iter_loss.reset()

        # 验证阶段
        model.eval()
        valid_epoch_loss = AverageMeter()
        valid_iter_loss = AverageMeter()
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(valid_loader):
                data, target = batch_samples['image'], batch_samples['label']
                data, target = Variable(data.to(device)), Variable(target.to(device))
                pred = model(data)
                loss = criterion(pred, target)
                image_loss = loss.item()
                valid_epoch_loss.update(image_loss)
                valid_iter_loss.update(image_loss)
                if batch_idx % iter_inter == 0:
                    logger.info('[val] epoch:{} iter:{}/{} {:.2f}% loss:{:.6f}'.format(
                        epoch, batch_idx, valid_loader_size, batch_idx / valid_loader_size * 100, valid_iter_loss.avg))
                    valid_iter_loss.reset()

        # 保存loss、lr
        train_loss_total_epochs.append(train_epoch_loss.avg)
        valid_loss_total_epochs.append(valid_epoch_loss.avg)
        epoch_lr.append(optimizer.param_groups[0]['lr'])
        # 保存模型
        if epoch % save_inter == 0 and epoch != 0:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(save_ckpt_dir, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)  # pytorch1.6会压缩模型，低版本无法加载
        # 保存最优模型
        if valid_epoch_loss.avg < best_loss:  # train_loss_per_epoch valid_loss_per_epoch
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(save_ckpt_dir, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_loss = valid_epoch_loss.avg
            best_mode = copy.deepcopy(model)
            logger.info('[save] Best Model saved at epoch:{} ============================='.format(epoch))
        scheduler.step()
        # 显示loss
        if epoch % disp_inter == 0: 
            logger.info('Epoch:{}, Training Loss:{:.6f}, Validation Loss:{:.6f}, Epoch_time:{}'.format(epoch, train_epoch_loss.avg, valid_epoch_loss.avg, second2time(time.time()-epoch_start)))

    # 训练loss曲线
    if plot:
        x = [i for i in range(epochs)]
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x, smooth(train_loss_total_epochs, 0.6), label='train loss')
        ax.plot(x, smooth(valid_loss_total_epochs, 0.6), label='val loss')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('CrossEntropy', fontsize=15)
        ax.set_title('train curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x, epoch_lr,  label='Learning Rate')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Learning Rate', fontsize=15)
        ax.set_title('lr curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()
            
    return best_mode, model

def normalize(img, mean, std, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img

def pred_image(model, image, target_l):
    image = normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = image.transpose(2, 0, 1)
    c, x, y = image.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    label = np.zeros((x, y))
    x_num = (x//target_l + 1) if x % target_l else x//target_l
    y_num = (y//target_l + 1) if y % target_l else y//target_l
    for i in range(x_num):
        for j in range(y_num):
            x_s, x_e = i*target_l, (i+1)*target_l
            y_s, y_e = j*target_l, (j+1)*target_l
            img = image[:, x_s:x_e, y_s:y_e]
            img = img[np.newaxis, :, :, :].astype(np.float32)
            img = torch.from_numpy(img)
            img = Variable(img.to(device))
            out_l = model(img)
            out_l = out_l.cpu().data.numpy()
            out_l = np.argmax(out_l, axis=1)[0]
            label[x_s:x_e, y_s:y_e] = out_l.astype(np.int8)
    # print(label.shape)
    return label


def cal_metrics(pred_label, gt):

    def _generate_matrix(gt_image, pre_image, num_class=2):
        mask = (gt_image >= 0) & (gt_image < num_class)#ground truth中所有正确(值在[0, classe_num])的像素label的mask
        label = num_class * gt_image[mask].astype('int') + pre_image[mask] 
        # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        count = np.bincount(label, minlength=num_class**2)
        confusion_matrix = count.reshape(num_class, num_class)#21 * 21(for pascal)
        return confusion_matrix

    def _Class_IOU(confusion_matrix):
        MIoU = np.diag(confusion_matrix) / (
                    np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                    np.diag(confusion_matrix))
        return MIoU

    confusion_matrix = _generate_matrix(gt.astype(np.int8), pred_label.astype(np.int8))
    miou = _Class_IOU(confusion_matrix)
    acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    return miou, acc