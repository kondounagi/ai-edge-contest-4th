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

import lib.transform_cv2 as T
from lib.sampler import RepeatedDistSampler
from lib.base_dataset_signate import BaseDataset, TransformationTrain, TransformationVal

#　こいつも冗長だし、evalの時も特にignoreとか生きていないみたいなので、一回削除
"""
labels_info = [
    {"hasInstances": False, "category": "void", "catid": 0, "name": "car", "ignoreInEval": True, "id": 0, "color": [0, 0, 255], "trainId": 0},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "bus", "ignoreInEval": True, "id": 1, "color": [193, 214, 0], "trainId": 1},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "truck", "ignoreInEval": True, "id": 2, "color": [180, 0, 129], "trainId": 2},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "svehicle", "ignoreInEval": True, "id": 3, "color": [255, 121, 166], "trainId": 3},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "pedestrian", "ignoreInEval": True, "id": 4, "color": [255, 0, 0], "trainId": 4},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "motorbike", "ignoreInEval": True, "id": 5, "color": [65, 166, 1], "trainId": 5},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "bicycle", "ignoreInEval": True, "id": 6, "color": [208, 149, 1], "trainId": 6},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "signal", "ignoreInEval": False, "id": 7, "color": [255, 255, 0], "trainId": 7},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "signs", "ignoreInEval": False, "id": 8, "color": [255, 134, 0], "trainId": 8},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "sky", "ignoreInEval": True, "id": 9, "color": [0, 152, 225], "trainId": 9},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "building", "ignoreInEval": True, "id": 10, "color": [0, 203, 151], "trainId": 10},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "natural", "ignoreInEval": False, "id": 11, "color": [85, 255, 50], "trainId": 11},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "wall", "ignoreInEval": False, "id": 12, "color": [92, 136, 125], "trainId": 12},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "lane", "ignoreInEval": False, "id": 13, "color": [69, 47, 142], "trainId": 13},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "ground", "ignoreInEval": True, "id": 14, "color": [136, 45, 66], "trainId": 14},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "sidewalk", "ignoreInEval": True, "id": 15, "color": [0, 255, 255], "trainId": 15},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "roadshoulder", "ignoreInEval": True, "id": 16, "color": [215, 0, 255], "trainId": 16},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "obstacle", "ignoreInEval": False, "id": 17, "color": [180, 131, 135], "trainId": 17},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "others", "ignoreInEval": True, "id": 18, "color": [82, 99, 0], "trainId": 18},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "own", "ignoreInEval": False, "id": 19, "color": [86, 62, 67], "trainId": 19},
]
"""


class Signate(BaseDataset):
    '''
    '''
    def __init__(self, root, resolution, num_class, trans_func=None, mode='train'):
        super(Signate, self).__init__(root, resolution, num_class, trans_func, mode)
        # self.n_cats = 20   # なんかここは変えなきゃな気がする。というかこいつ使われてないんだけど何？
        self.lb_ignore = -1
        # なくていい気がするから、一回むし。というか、base_datasetとやってることが被りがちなのでよくない？
        """
        self.lb_map = np.arange(256).astype(np.uint8)
        for el in labels_info:
            self.lb_map[el['id']] = el['trainId'] # 今回はel[i] = i
        """
        self.to_tensor = T.ToTensor(
            mean=(0.3744, 0.3841, 0.4029), # signateに合わせた。cityscape も事前にこれに合わせるのでスイッチはいらない
            std=(0.2778, 0.2752, 0.2596),
        )


def get_data_loader(root, resolution, num_class, ims_per_gpu, scales, cropsize, max_iter=None, mode='train', distributed=False):
    print("in get_data_loader")
    if mode == 'train':
        trans_func = TransformationTrain(scales, cropsize)
        batchsize = ims_per_gpu
        shuffle = True
        drop_last = True
    elif mode == 'val':
        trans_func = TransformationVal()
        batchsize = ims_per_gpu
        shuffle = False
        drop_last = False

    ds = Signate(root, resolution, num_class, trans_func=trans_func, mode=mode)

    if distributed:
        assert dist.is_available(), "dist should be initialzed"
        if mode == 'train':
            assert not max_iter is None
            n_train_imgs = ims_per_gpu * dist.get_world_size() * max_iter
            sampler = RepeatedDistSampler(ds, n_train_imgs, shuffle=shuffle)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                ds, shuffle=shuffle)
        batchsampler = torch.utils.data.sampler.BatchSampler(
            sampler, batchsize, drop_last=drop_last
        )
        dl = DataLoader(
            ds,
            batch_sampler=batchsampler,
            num_workers=4,
            pin_memory=True,
        )
    else:
        dl = DataLoader(
            ds,
            batch_size=batchsize,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=4,
            pin_memory=True,
        )
    return dl



if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    ds = Signate('./data/', mode='val')
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
