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
from rich.progress import track


_valid_class = [29, 64, 69, 70, 75, 76, 82, 93, 115, 117,
                122, 136, 146, 150, 155, 166, 179, 181, 183, 226]
pix2index = np.ones(256, dtype=int) * -1
pix2index[_valid_class] = range(20) # 上のsignate_pallette_indexと同じ配列。ただ、ignored classを−１にした。

# クラスすうによってまとめるものが違うから。
# signateの評価指標に使うものに0~3を割り当て
as_same_class5 = np.array([-1, 0, 1, 4, 4, 4, 2, 4, 4, 4, 4,
                            4, 4, 4, 4, 4, 4, 4, 4, 4, 3])
as_same_class14 = np.array([-1, 0, 1, 5, 6, 4, 2, 6, 7, 8, 9,
                            10, 11, 12, 9, 13, 5, 7, 4, 5, 3])
as_same_class19 = np.array([-1, 0, 1, 5, 6, 4, 2, 6, 7, 8, 9,
                            10, 11, 12, 13, 14, 15, 16, 17, 18, 3])
as_same_class = {5 : as_same_class5,
                14 : as_same_class14,
                19 : as_same_class19}

class BaseDataset(Dataset):
    def __init__(self, root, resolution, num_class, trans_func=None, mode='train', dataset='cityscapes'):
        super(BaseDataset, self).__init__()
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.dataset = dataset # ここでcityscapesがデフォになってるけど、そのままだと、マジで今まで通りになるので、忘れても事故は起きない
        self.trans_func = trans_func
        self.resolution = resolution
        self.as_same_class = as_same_class[num_class]


        # txtファイルからpairを吐かせるという仕様（激おこ）だったので、画像ディレクトリのパスを
        # 渡せば、imgとmaskを返してくれるようにした。
        # pathとpth混合すな        self.img_paths, self.lb_paths = _get_paired_img_path(root)
        #print(self.lb_paths)
        self.img_paths, self.lb_paths = _get_paired_img_path(root)
        if self.mode != 'test':
            assert len(self.img_paths) == len(self.lb_paths)
        self.len = len(self.img_paths)
        self.imgs = []
        self.lbs = []
        self._load_on_memory()

    def _load_on_memory(self):
        for i in track(range(self.len), description='loading images'):
            img = cv2.imread(self.img_paths[i])
            lb = np.asarray(Image.open(self.lb_paths[i]).convert('L'))
            self.imgs.append(img)
            self.lbs.append(lb)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        lb = self.lbs[idx]
        img, lb = self._resize(img, lb)
        lb = pix2index[lb]
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
        # cityscapesの場合と、signateの場合でアスペクト比が違うのでそのための場合わけ.
        # わざわざ、データセットで指定しないのは、同じ実験でもevalのときはsignateを使うから 
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
        if 'png' in filename or 'jpg' in filename:
            img_paths[i] = os.path.join(root, img_root, filename)
    for i, filename in enumerate(lb_paths):
        if 'png' in filename or 'jpg' in filename:
            lb_paths[i] = os.path.join(root, lb_root, filename)
    img_paths.sort()
    lb_paths.sort()

    return img_paths, lb_paths



class TransformationTrain(object):

    def __init__(self, scales, cropsize, dataset='cityscapes'):
        if dataset == 'signate' or dataset == 'cityscapes':
            self.trans_func = T.Compose([
                T.RandomResizedCrop(scales, cropsize),
                T.RandomHorizontalFlip(),
                T.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4
                ),
            ])
        elif dataset == 'cityscapes_night':
            self.trans_func = T.Compose([
                T.RandomResizedCrop(scales, cropsize),
                T.RandomHorizontalFlip(),
                T.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4
                ),
                T.RandomNightBrightness(), # ここで、めっちゃ輝度落とす
            ])
        else:
            raise ValueError('cityscapes or signate dataset only')
            

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
