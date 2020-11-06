import sys
sys.path.insert(0, '.')
import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
from PIL import Image
import numpy as np
import cv2

from lib.models import model_factory
from configs import cfg_factory
from lib.base_dataset import as_same_class
from lib.color_palette import get_palette
from rich.progress import track
from lib.signate_cv2 import get_data_loader

import re
from rich.progress import track

#torch.set_grad_enabled(False) こいつクソ害悪だろ
np.random.seed(123)

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--exp', type=str, default='logs/res_2020_11_01_17_56', help='which experiment do you test ?') # res/res_mm_dd_..のやつを指定して
    parse.add_argument('--save_folder', type=str, default='res/images') # 画像を吐かせる先
    parse.add_argument('--model', dest='model', type=str, default='bisenetv2_light')
    parse.add_argument('--num_class', type=int, default=13)
    parse.add_argument('--root', type=str, default='datasets/test')
    parse.add_argument('--mode', type=str, default='test', choices=['val', 'test'])
    parse.add_argument('--resolution', type=int, default=1024)

    return parse.parse_args()

def get_test_loader(root, resolution): # 不覚。imとlbを同時に吐く仕様は変えてなかった...（そのおかげで、コードの簡単になっているところもあるのだけれど...)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4029, 0.3841, 0.3744), (0.2596, 0.2752, 0.2778))
    ])

    img_paths = os.listdir(root)
    img_paths.sort()
    regex = re.compile('\d+')
    edge_num = regex.findall(img_paths[0])[0]

    imgs = []
    for img_path in img_paths:
        img_path = os.path.join(root, img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (resolution, resolution * 5 // 8)) # ごめん、ハードコード。appの方でもresizeはcv2が使われているので、ここでもcv2　cv2 は短いほうが後だよ
        img = transform(img) # ここで変換
        imgs.append(img)
    return imgs, img_paths

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    print(args)
    if not os.path.exists(args.save_folder): os.makedirs(args.save_folder)
    #args.root = os.path.join(args.root, '{}'.format(time.strftime('%Y_%m_%d_%H_%M')))
    dl, img_paths = get_test_loader(args.root, args.resolution)
    dl = DataLoader(dl, batch_size=1, shuffle=False)

    cfg = cfg_factory['bisenetv2']

    palette = get_palette(args.num_class)

    # define model
    net = model_factory['bisenetv2_light'](args.num_class)
    net.load_state_dict(torch.load(os.path.join(args.exp, 'model_final.pth')), strict=False)
    net.eval()
    net.to(device)

    for i, img in track(enumerate(dl), total=len(dl)):
        with torch.no_grad():
            img = img.to(device)
            out = net(img).argmax(dim=1).squeeze().cpu().numpy()
            pred = palette[out]
            pred = cv2.resize(pred, (1936, 1216), interpolation=cv2.INTER_NEAREST) 
            wo_ext, _ = os.path.splitext(img_paths[i])
            cv2.imwrite(os.path.join(args.save_folder, wo_ext + '.png'), pred)
            


if __name__ =='__main__':
    main()
