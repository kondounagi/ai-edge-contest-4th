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
""" ごめん、これデバッグがえぐくなりがちなので変えた。
signate_pallette = [29, 183, 69, 166, 76,
                    117, 150, 226, 155, 115,
                    136, 181, 122,  64,  75,
                    179,  93, 146,  82,  70]
signate_pallette_index = np.array([255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255,   0, 255, 255, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,  13,
       255, 255, 255, 255,   2, 18, 255, 255, 255, 255,  14,   4, 255,
       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
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
"""
_valid_class = [29, 64, 69, 70, 75, 76, 82, 93, 115, 117,
                122, 136, 146, 150, 155, 166, 179, 181, 183, 226]
pix2index = np.ones(256, dtype=int) * -1
pix2index[_valid_class] = range(20) # 上のsignate_pallette_indexと同じ配列。ただ、ignored classを−１にした。

# クラスすうによってまとめるものが違うから。
as_same_class5 = np.array([-1, 0, 1, 4, 4, 4, 2, 4, 4, 4, 4,
                            4, 4, 4, 4, 4, 4, 4, 4, 4, 3])
as_same_class13 = np.array([-1, 0, 1, 2, 3, 4, 5, 3, 4, 8, 9,
                            10, 11, 10, 9, 6, 2, 4, 7, 2, 12])
as_same_class20 = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                            10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
as_same_class = {5 : as_same_class5,
                13 : as_same_class13,
                20 : as_same_class20}

class BaseDataset(Dataset):
    '''
    '''
    def __init__(self, root, resolution, num_class, trans_func=None, mode='train'):
        super(BaseDataset, self).__init__()
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.trans_func = trans_func
        self.resolution = resolution
        self.as_same_class = as_same_class[num_class]

        self.lb_map = None # こいつは使い方を大きく変えたので注意。

        # txtファイルからpairを吐かせるという仕様（激おこ）だったので、画像ディレクトリのパスを
        # 渡せば、imgとmaskを返してくれるようにした。
        # pathとpth混合すな。
        """
        with open(ann_root, 'r') as fr:
            pairs = fr.read().splitlines()
        self.img_paths, self.lb_paths = [], []
        for pair in pairs:
            imgpth, lbpth = pair.split(',')
            self.img_paths.append(osp.join(img_root, imgpth))
            self.lb_paths.append(osp.join(img_root, lbpth))
        """
        self.img_paths, self.lb_paths = _get_paired_img_path(root)
        #print(self.lb_paths)

        assert len(self.img_paths) == len(self.lb_paths)
        self.len = len(self.img_paths)

    def __getitem__(self, idx):
        img_path, lb_path = self.img_paths[idx], self.lb_paths[idx]
        #img, label = cv2.imread(impth), cv2.imread(lbpth, 0)
        img = cv2.imread(img_path) 
        lb = np.asarray(Image.open(lb_path).convert('L')) # さっきはここまで書いた。
        img, lb = self._resize(img, lb)
        """
        if not self.lb_map is None:
            lb = self.lb_map[lb] 
        """
        #convert from signate grayscale value to labels
        #lb = signate_pallette_index[lb] 
        #print('before map', np.unique(lb))
        lb = pix2index[lb]
        #print('after map', np.unique(lb))
        lb = self.as_same_class[lb + 1] # クラス数に応じてまとめられる。lb+1は-1のignore idxを考慮して。
        im_lb = dict(im=img, lb=lb)
        if not self.trans_func is None:
            im_lb = self.trans_func(im_lb)
        im_lb = self.to_tensor(im_lb)
        img, lb = im_lb['im'], im_lb['lb']
        return img.detach(), lb.unsqueeze(0).detach()

    def __len__(self):
        return self.len

    def _resize(self, img, lb):
        # cityscapesの場合と、signateの場合でアスペクト比が違うのでそのための場合わけ。
        if img.shape[1] / img.shape[0] == 2:
            img = cv2.resize(img, (self.resolution, self.resolution // 2))
            lb = cv2.resize(lb, (self.resolution, self.resolution // 2), interpolation=cv2.INTER_NEAREST)
        elif 1 < img.shape[1] / img.shape[0] < 2:
            img = cv2.resize(img, (self.resolution, self.resolution * 5 // 8))
            lb = cv2.resize(lb, (self.resolution, self.resolution *5 // 8), interpolation=cv2.INTER_NEAREST)
        else:
            raise ValueError('cityscapes or signate dataset only')
        return img, lb


# i番目のimgとlbがちゃんと対応するようにpathの集合（つまりまだ文字列）を返す。
def _get_paired_img_path(root):
    img_root = 'img'
    lb_root = 'lb'
    img_paths = os.listdir(os.path.join(root, img_root))
    lb_paths = os.listdir(os.path.join(root, lb_root))
    for i, filename in enumerate(img_paths):
        img_paths[i] = os.path.join(root, img_root, filename)
    for i, filename in enumerate(lb_paths):
        lb_paths[i] = os.path.join(root, lb_root, filename)
    img_paths.sort()
    lb_paths.sort()

    return img_paths, lb_paths



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
