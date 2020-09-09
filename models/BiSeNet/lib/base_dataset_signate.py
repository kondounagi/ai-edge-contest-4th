#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import os.path as osp
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import cv2
import numpy as np
from PIL import Image

import lib.transform_cv2 as T
from lib.sampler import RepeatedDistSampler

signate_pallette = [29, 183, 69, 166, 76,
                    117, 150, 226, 155, 115,
                    136, 181, 122,  64,  75,
                    179,  93, 146,  82,  70]
signate_pallette_index = np.array([255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255,   0, 255, 255, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,  13,
       255, 255, 255, 255,   2, 255, 255, 255, 255, 255,  14,   4, 255,
       255, 255, 255, 255,  18, 255, 255, 255, 255, 255, 255, 255, 255,
       255, 255,  16, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   9, 255,
         5, 255, 255, 255, 255,  12, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255, 255, 255, 255,  10, 255, 255, 255, 255, 255, 255,
       255, 255, 255,  17, 255, 255, 255,   6, 255, 255, 255, 255,   8,
       255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   3, 255, 255,
       255, 255, 255, 255, 255, 255, 255, 255, 255, 255,  15, 255,  11,
       255,   1, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255, 255, 255,   7, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255, 255, 255, 255, 255, 255])
class BaseDataset(Dataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(BaseDataset, self).__init__()
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.trans_func = trans_func

        self.lb_map = None

        with open(annpath, 'r') as fr:
            pairs = fr.read().splitlines()
        self.img_paths, self.lb_paths = [], []
        for pair in pairs:
            imgpth, lbpth = pair.split(',')
            self.img_paths.append(osp.join(dataroot, imgpth))
            self.lb_paths.append(osp.join(dataroot, lbpth))
        #print(self.img_paths)
        #print(self.lb_paths)

        assert len(self.img_paths) == len(self.lb_paths)
        self.len = len(self.img_paths)

    def __getitem__(self, idx):
        impth, lbpth = self.img_paths[idx], self.lb_paths[idx]
        #img, label = cv2.imread(impth), cv2.imread(lbpth, 0)
        img, label = cv2.imread(impth), np.asarray(Image.open(lbpth).convert("L"))
        if not self.lb_map is None:
            label = self.lb_map[label]
        
        #convert from signate grayscale value to labels
        label = signate_pallette_index[label]
        im_lb = dict(im=img, lb=label)
        if not self.trans_func is None:
            im_lb = self.trans_func(im_lb)
        im_lb = self.to_tensor(im_lb)
        img, label = im_lb['im'], im_lb['lb']
        return img.detach(), label.unsqueeze(0).detach()

    def __len__(self):
        return self.len


class TransformationTrain(object):

    def __init__(self, scales, cropsize):
        self.trans_func = T.Compose([
            T.RandomResizedCrop(scales, cropsize),
            T.RandomHorizontalFlip(),
            T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4
            ),
        ])

    def __call__(self, im_lb):
        im_lb = self.trans_func(im_lb)
        return im_lb


class TransformationVal(object):

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        return dict(im=im, lb=lb)


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    ds = CityScapes('./data/', mode='val')
    dl = DataLoader(ds,
                    batch_size = 4,
                    shuffle = True,
                    num_workers = 4,
                    drop_last = True)
    for imgs, label in dl:
        print(len(imgs))
        for el in imgs:
            print(el.size())
        break
