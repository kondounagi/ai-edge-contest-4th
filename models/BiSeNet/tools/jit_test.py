
import os
import re
import sys
sys.path.insert(0, '.')
import argparse
import time
import pdb
import random

import torch
import torchvision
import torchvision.transforms as transforms


from lib.models.bisenetv2 import BiSeNetV2_Light
from lib.signate_cv2 import get_data_loader
from tools.eval import eval_model

from tqdm import tqdm

def parse_args():
  parse = argparse.ArgumentParser()
  parse.add_argument('--weight_path', type=str, default='logs/res_2020_11_08_23_05/model_156999.pth')
  parse.add_argument('--resolution', type=int, default=1024)
  parse.add_argument('--num_class', type=int, default=14)
  parse.add_argument('--train_root', type=str, default='../datasets/finetune/train')
  parse.add_argument('--val_root', type=str, default='../datasets/finetune/val')
  return parse.parse_args()


if __name__ == '__main__':
  _args = parse_args()
  file_path = _args.weight_path

  model = BiSeNetV2_Light(_args.num_class).cpu() # __init__はさすがにあっておっけいじゃないと困る
  model.load_state_dict(torch.load(file_path), strict=False)
  model.eval()

  random_input = torch.rand(1, 3, 640, 1024)
  
  traced_net = torch.jit.trace(model, random_input)
  traced_net.save('bisenetv2_light_jit.pt')


