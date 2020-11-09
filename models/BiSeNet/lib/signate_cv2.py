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
from lib.base_dataset import BaseDataset, TransformationTrain, TransformationVal


class Signate(BaseDataset):
    def __init__(self, root, resolution, num_class, trans_func=None, mode='train', dataset='cityscapes'):
        super(Signate, self).__init__(root, resolution, num_class, trans_func, mode, dataset=dataset)
        self.lb_ignore = -1
        self.dataset = dataset
        self.to_tensor = T.ToTensor(
            mean=(0.4029, 0.3841, 0.3744), # signateに合わせた。cityscape も事前にこれに合わせるのでスイッチはいらない
            std=(0.2596, 0.2752, 0.2778),
        )
 

def get_data_loader(root, resolution, num_class, ims_per_gpu, scales, cropsize, max_iter=None, mode='train', distributed=False, dataset='cityscapes'):
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

    ds = Signate(root, resolution, num_class, trans_func=trans_func, mode=mode, dataset=dataset)

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
