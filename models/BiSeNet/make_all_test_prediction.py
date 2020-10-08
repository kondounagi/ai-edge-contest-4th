import sys
sys.path.insert(0, '.')
import os

import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import time
from PIL import Image
import numpy as np
import cv2

from lib.models import model_factory
from configs import cfg_factory

torch.set_grad_enabled(False)
np.random.seed(123)


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



# args
parse = argparse.ArgumentParser()
parse.add_argument('--model', dest='model', type=str, default='bisenetv2',)
parse.add_argument('--weight-path', type=str, default='./res/model_2020_09_12_14_56.pth',)
parse.add_argument('--in-dir-path', dest='in_dir_path', type=str)
parse.add_argument('--out-dir-path', dest='out_dir_path', type=str)
args = parse.parse_args()
cfg = cfg_factory[args.model]


# palette and mean/std
#palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
palette = []
for i in range(len(labels_info)):
    if labels_info[i]["trainId"] == -1 or labels_info[i]["trainId"] == 255:
        pass
    else:
        palette.append(labels_info[i]["color"])
palette = np.array(palette, dtype=np.uint8)
#print(palette)
mean = torch.tensor([0.3744, 0.3841, 0.4029], dtype=torch.float32).view(-1, 1, 1)
std = torch.tensor([0.2778, 0.2752, 0.2596], dtype=torch.float32).view(-1, 1, 1)

# define model
net = model_factory[cfg.model_type](19)
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
net.eval()
net.cuda()


in_dir = os.listdir(args.in_dir_path)
for img_name in in_dir:
    print("read: ", img_name)
    # prepare data
    print(os.path.join(args.in_dir_path, img_name))
    img = cv2.imread(os.path.join(args.in_dir_path, img_name))
    img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
    img = torch.from_numpy(img).div_(255).sub_(mean).div_(std).unsqueeze(0).cuda()

    # inference
    print("imference start")
    out = net(img)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
    pred = palette[out]
    pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
    #cv2.imwrite('./res/image/res_{}.png'.format(time.strftime('%Y_%m_%d_%H_%M')), pred)
    root, ext = os.path.splitext(img_name)
    out_name = root + ".png"
    cv2.imwrite(os.path.join(args.out_dir_path, out_name), pred)
    print("write: ", os.path.join(args.out_dir_path, out_name))