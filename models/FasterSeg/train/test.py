#!/usr/bin/env python3
# encoding: utf-8
import os
import time
import cv2
cv2.setNumThreads(0)
import torchvision
from PIL import Image
import argparse
import numpy as np

import torch
import torch.multiprocessing as mp

from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_prediction
from engine.tester import Tester
from engine.logger import get_logger
from seg_opr.metric import hist_info, compute_score
from datasets.cityscapes import Cityscapes

logger = get_logger()


cityscapes_trainID2id = {
  0: 7,
  1: 8,
  2: 11,
  3: 12,
  4: 13,
  5: 17,
  6: 19,
  7: 20,
  8: 21,
  9: 22,
  10: 23,
  11: 24,
  12: 25,
  13: 26,
  14: 27,
  15: 28,
  16: 31,
  17: 32,
  18: 33,
  19: 0
}

class SegTester(Tester):
    def func_per_iteration(self, data, device, iter=None):
        if self.config is not None: config = self.config
        img = data['data']
        name = data['fn']
        name = name.cpu().numpy()
        print("name = ", name)

        if len(config.eval_scale_array) == 1:
            # pred: bhw
            pred = self.whole_eval(img, None, device)
        else:
            pred = self.sliding_eval(img, config.eval_crop_size, config.eval_stride_rate, device)
            
        # interpolate
        pred_resized = np.zeros([0, 1216, 1936], dtype=np.uint8)
        for i in range(pred.shape[0]):
            temp = cv2.resize(pred[i], (1936,1216), interpolation=cv2.INTER_NEAREST)
            pred_resized = np.append(pred_resized, np.expand_dims(temp, axis=0), axis=0)
            
        image = img.cpu().numpy()
        image_resized = np.zeros([0, 3, 1216, 1936], dtype=np.uint8)
        for i in range(image.shape[0]):
            temp = image[i].transpose(1,2,0)
            temp = cv2.resize(temp, (1936,1216), interpolation=cv2.INTER_NEAREST)
            temp = temp.transpose(2,0,1)
            image_resized = np.append(image_resized, np.expand_dims(temp, axis=0), axis=0)

        if self.show_prediction:
            colors = self.dataset.get_class_colors()
            image = img.cpu().numpy()
            # comp_img: hwc, rgb
            comp_img = show_prediction(colors, config.background, image_resized, pred_resized)
            
            for i in range(len(name)):
                cv2.imwrite(os.path.join(os.path.realpath('.'), self.config.save, "test", "test_"+str(name[i][0]).zfill(3)+".png"), comp_img[i,:,:,::-1])

#         for x in range(pred.shape[0]):
#             for y in range(pred.shape[1]):
#                 pred[x, y] = cityscapes_trainID2id[pred[x, y]]
#         cv2.imwrite(os.path.join(os.path.realpath('.'), self.config.save, "test", name+".png"), pred)

    def compute_metric(self, results):
        hist = np.zeros((self.config.num_classes, self.config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iu, mean_IU, mean_IU_no_back, mean_pixel_acc = compute_score(hist, correct, labeled)
        result_line = print_iou(iu, mean_pixel_acc, self.dataset.get_class_names(), True)
        return result_line, mean_IU
