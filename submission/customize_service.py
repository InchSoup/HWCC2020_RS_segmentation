# -*- coding: utf-8 -*-
from collections import OrderedDict
from efficientunet import *
# from unet import UNet
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from model_service.pytorch_model_service import PTServingBaseService
import cv2
import time
# from metric.metrics_manager import MetricsManager
import log
from io import BytesIO
import base64
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000000000
logger = log.getLogger(__name__)


class ImageClassificationService(PTServingBaseService):
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path
        self.model = get_efficientunet_b3(out_channels=2, concat_input=True, pretrained=False)
        self.model_1 = get_efficientunet_b4(out_channels=2, concat_input=True, pretrained=False)
        self.use_cuda = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print('Using GPU for inference')
            self.use_cuda = True
            checkpoint = torch.load(self.model_path)
            self.model = self.model.to(device)
            self.model.load_state_dict(checkpoint['b3_state_dict']['state_dict'])
            self.model_1 = self.model_1.to(device)
            self.model_1.load_state_dict(checkpoint['b4_state_dict']['state_dict'])
        else:
            print('Using CPU for inference')
            checkpoint = torch.load(self.model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['b3_state_dict']['state_dict'])
            self.model_1.load_state_dict(checkpoint['b4_state_dict']['state_dict'])

        self.model.eval()

    def _normalize(self, img, mean, std, max_pixel_value=255.0):
        mean = np.array(mean, dtype=np.float32)
        mean *= max_pixel_value

        std = np.array(std, dtype=np.float32)
        std *= max_pixel_value

        denominator = np.reciprocal(std, dtype=np.float32)

        img = img.astype(np.float32)
        img -= mean
        img *= denominator
        return img
    def _test_augment(self, image):
        # img = np.array(image)
        # img = img.squeeze()
        img = image.transpose(1, 2, 0)
        # img = cv2.resize(img, image_size)

        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)

        return torch.from_numpy(img5)

    def _test_augment_pred(self, pred):
        pred_1 = pred[:,0,:,:].squeeze()
        pred_2 = pred[:,1,:,:].squeeze()
        #0channel
        pred1_1 = pred_1[:4] + pred_1[4:, :, ::-1]
        pred2_1 = pred1_1[:2] + pred1_1[2:, ::-1]
        pred3_1 = pred2_1[0] + np.rot90(pred2_1[1])[::-1, ::-1]
        pred_1 = pred3_1.copy()/8.
        #1channel
        pred1_2 = pred_2[:4] + pred_2[4:, :, ::-1]
        pred2_2 = pred1_2[:2] + pred1_2[2:, ::-1]
        pred3_2 = pred2_2[0] + np.rot90(pred2_2[1])[::-1, ::-1]
        pred_2 = pred3_2.copy()/8.

        pred_out = np.array([pred_1,pred_2])
        return pred_out
    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = Image.open(file_content)
                # img = self.transforms(img)
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                preprocessed_data[k] = img
        return preprocessed_data

    def _inference(self, data):
        img = data["input_img"]
        data = img
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = self._normalize(data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        data = data.transpose(2, 0, 1)
        l = 512    # 推理图片尺寸
        b = 72     # 忽略边缘像素大小
        a = l-2*b  # 实际提取的图片尺寸
        
        c, H, W = data.shape
        label = np.zeros((H, W))
        W_num = (W-(l-b))//a+2 if (W-(l-b))//a else (W-(l-b))//a+1
        H_num = (H-(l-b))//a+2 if (H-(l-b))//a else (H-(l-b))//a+1

        for i in range(W_num):
            for j in range(H_num):
                x_min = l-2*b + (i-1)*a if i>0 else 0
                x_max = x_min+l
                
                y_min = l-2*b + (j-1)*a if j>0 else 0
                y_max = y_min+l  
            
                # 越界的情况
                if x_max > W:
                    x_max = W
                    x_min = W-l

                    if xx_max!=W:
                        xx_min = xx_max
                    xx_max = x_max

                else:
                    xx_min = x_min+b
                    xx_max = x_max-b
        
                if y_max > H:
                    y_max = H
                    y_min = H-l

                    yy_min = yy_max
                    yy_max = y_max

                else:
                    yy_min = y_min+b
                    yy_max = y_max-b

                # 初始的情况

                if i==0:
                    xx_min = 0  

                if j==0:
                    yy_min = 0
                img = data[:, y_min:y_max, x_min:x_max]
                
                # img = img[np.newaxis, :, :, :].astype(np.float32)
                # img = torch.from_numpy(img)
                # img = Variable(img.to(device))

                image_augment = self._test_augment(img)
                image_augment = Variable(image_augment.to(device))
                with torch.no_grad():
                    output_augment = self.model(image_augment)
                    output_augment_1 = self.model_1(image_augment)
                    output_augment = (output_augment+output_augment_1)/2.
                pred_augment = output_augment.squeeze().cpu().data.numpy()
                out_l = self._test_augment_pred(pred_augment)
                # out_l = self.model(img)
                # out_l = out_l.cpu().data.numpy()
                out_l = np.argmax(out_l, axis=0)
                label[yy_min:yy_max, xx_min:xx_max] = out_l[yy_min-y_min:yy_max-y_min, xx_min-x_min:xx_max-x_min].astype(np.int8) #  提取图片相对位置
        # _label = label.astype(np.int8).tolist()
        _label = label.astype(np.int8).tolist()
        _len, __len = len(_label), len(_label[0])
        o_stack = []
        for _ in _label:
            out_s = {"s":[], "e":[]}
            j = 0
            while j < __len:
                if _[j] == 0:
                    out_s["s"].append(str(j))
                    while j < __len and _[j] == 0: j += 1
                    out_s["e"].append(str(j))
                j += 1
            o_stack.append(out_s)
        result = {"result": o_stack}
        return result

    def _postprocess(self, data):
        return data

    def inference(self, data):
        pre_start_time = time.time()
        data = self._preprocess(data)
        infer_start_time = time.time()
        # Update preprocess latency metric
        pre_time_in_ms = (infer_start_time - pre_start_time) * 1000
        logger.info('preprocess time: ' + str(pre_time_in_ms) + 'ms')
        # if self.model_name + '_LatencyPreprocess' in MetricsManager.metrics:
        #     MetricsManager.metrics[self.model_name + '_LatencyPreprocess'].update(pre_time_in_ms)
        data = self._inference(data)
        infer_end_time = time.time()
        infer_in_ms = (infer_end_time - infer_start_time) * 1000
        logger.info('infer time: ' + str(infer_in_ms) + 'ms')
        data = self._postprocess(data)
        # Update inference latency metric
        post_time_in_ms = (time.time() - infer_end_time) * 1000
        logger.info('postprocess time: ' + str(post_time_in_ms) + 'ms')
        # if self.model_name + '_LatencyInference' in MetricsManager.metrics:
        #     MetricsManager.metrics[self.model_name + '_LatencyInference'].update(post_time_in_ms)
        # Update overall latency metric
        # if self.model_name + '_LatencyOverall' in MetricsManager.metrics:
        #     MetricsManager.metrics[self.model_name + '_LatencyOverall'].update(pre_time_in_ms + post_time_in_ms)
        logger.info('latency: ' + str(pre_time_in_ms + infer_in_ms + post_time_in_ms) + 'ms')
        data['latency_time'] = pre_time_in_ms + infer_in_ms + post_time_in_ms
        return data