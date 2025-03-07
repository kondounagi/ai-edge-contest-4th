
import sys
sys.path.insert(0, '.')
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
from lib.base_dataset import as_same_class
from lib.color_palette import get_palette

#torch.set_grad_enabled(False) こいつクソ害悪だろ
np.random.seed(123)

def main():
    # args
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', dest='model', type=str, default='bisenetv2',)
    parse.add_argument('--weight-path', type=str, default='./res/model_final.pth',)
    parse.add_argument('--img-path', dest='img_path', type=str, default='./example.png',)
    parse.add_argument('--num_class', type=int, default=13)
    args = parse.parse_args()
    cfg = cfg_factory[args.model]


    # palette and mean/std
    #palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

    palette = get_palette(args.num_class)
    print(palette)
    # こいつの順番入れ替えてね. cv2なのでBGR
    #mean = torch.tensor([0.3744, 0.3841, 0.4029], dtype=torch.float32).view(-1, 1, 1)
    #std = torch.tensor([0.2778, 0.2752, 0.2596], dtype=torch.float32).view(-1, 1, 1)

    # define model
    net = model_factory[cfg.model_type](19)
    net.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
    net.eval()
    net.cuda()

    # prepare data
    im = cv2.imread(args.img_path)
    im = im[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
    im = torch.from_numpy(im).div_(255).sub_(mean).div_(std).unsqueeze(0).cuda()

    # inference
    with torch.no_grad():
        out = net(im)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
        print(out)
        pred = palette[out]
        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        cv2.imwrite('./res/image/res_{}.png'.format(time.strftime('%Y_%m_%d_%H_%M')), pred)

if __name__ =='__main__':
    main()
