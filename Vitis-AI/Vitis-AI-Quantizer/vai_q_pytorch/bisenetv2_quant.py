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

from pytorch_nndct.apis import torch_quantizer, dump_xmodel # こいつが肝

from lib.models.bisenetv2 import BiSeNetV2_Light
from lib.signate_cv2 import get_data_loader
from tools.eval import eval_model

from tqdm import tqdm

def parse_args():
  parse = argparse.ArgumentParser()
  parse.add_argument('--weight-path', type=str,
                     default='./pths/bisenetv2_model_2020_09_12_14_48.pth',
                     required=True)
  parse.add_argument('--resolution', type=int, default=1024)
  parse.add_argument('--num_class', type=int, default=19)
  parse.add_argument('--train_root', type=str, default='../datasets/finetune/train')
  parse.add_argument('--val_root', type=str, default='../datasets/finetune/val')
  parse.add_argument('--quant-mode', type=int, default=1)
  parse.add_argument('--imgs-per-gpu', type=int, default=8)
  args = parse.parse_args()
  return args


def quantization(title='optimize', model_name='', file_path='', quant_mode=1):
  print('quantization start')
  batch_size = args.imgs_per_gpu

  model = BiSeNetV2_Light(args.num_class).cpu() # __init__はさすがにあっておっけいじゃないと困る
  print('model was loaded on cpu')
  model.load_state_dict(torch.load(file_path), strict=False)
  print('weight file was loaded on cpu')
  print(model)

  input = torch.randn([batch_size, 3, args.resolution * 5 // 8, args.resolution]) # どっちがどっちだかわかんないけど
  if quant_mode < 1:
    quant_model = model
  else:
    ## new api
    ####################################################################################
    print('start torch_quantizer')

    quantizer = torch_quantizer(
        quant_mode, model, (input))

    print('end torch_quantizer')

    quant_model = quantizer.quant_model
    #####################################################################################

  # to get loss value after evaluation
  # calibrationの部分がどこにあるのか分からんけどここをいじれば良さそう
  # とにかくなんか精度が出ればいいんですかね
  # record  modules float model accuracy
  # add modules float model accuracy here

  #register_modification_hooks(model_gen, train=False)
  heads, mious = eval_model(model, args.ims_per_gpu, args.val_root, args.resolution, args.num_class)

  # logging accuracy
  print('mious: ', mious) # こんなんでいいんすかね
  #print('loss: %g' % (loss_gen))
  #print('top-1 / top-5 accuracy: %g / %g' % (acc1_gen, acc5_gen))

  # ここはほっとくけど、大事なところ
  if quant_mode > 0:
    quantizer.export_quant_config()
    if quant_mode == 2:
      dump_xmodel()

if __name__ == '__main__':
  args = parse_args()
  file_path = args.weight_path
  model_name = 'bisenetv2_light'

  feature_test = ' float model evaluation'
  if args.quant_mode > 0:
    feature_test = ' quantization'
    # force to merge BN with CONV for better quantization accuracy
    args.optimize = 1
    feature_test += ' with optimization'
  else:
    feature_test = ' float model evaluation'
  title = model_name + feature_test

  print("-------- Start {} test ".format(model_name))

  # calibration or evaluation
  quantization(
      title=title,
      model_name=model_name,
      file_path=file_path,
      quant_mode=args.quant_mode)

  print("-------- End of {} test ".format(model_name))

