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

from lib.model.bisenetv2 import BiseNetV2_Light
from lib.signate_cv2 import get_data_loader
from tools.eval import eval_model

from tqdm import tqdm

def parse_args():
  parse = argparse.ArgumentParser()
  parse.add_argument('--weight_path', type=str, default='../logs/res_2020_dd_hh_mm')
  parse.add_argument('--resolution', type=int, default=1024)
  parse.add_argument('--num_class', type=int, default=13)
  parse.add_argument('--train_root', type=str, default='../datasets/finetune/train')
  parse.add_argument('--val_root', type=str, default='../datasets/finetune/val')


def quantization(title='optimize', model_name='', file_path='', quant_mode=1):
  batch_size = _args.ims_per_gpu

  model = BiseNetV2_Light(_args.num_class).cpu() # __init__はさすがにあっておっけいじゃないと困る
  model.load_state_dict(torch.load(file_path), strict=False)

  input = torch.randn([batch_size, 3, _args.resolution * 5 // 8, _args.resolution]) # どっちがどっちだかわかんないけど
  if quant_mode < 1:
    quant_model = model
  else:
    ## new api
    ####################################################################################
    quantizer = torch_quantizer(
        quant_mode, model, (input))

    quant_model = quantizer.quant_model
    #####################################################################################

  # to get loss value after evaluation
  # calibrationの部分がどこにあるのか分からんけどここをいじれば良さそう
  # とにかくなんか精度が出ればいいんですかね
  # record  modules float model accuracy
  # add modules float model accuracy here

  #register_modification_hooks(model_gen, train=False)
  heads, mious = eval_model(model, _args.ims_per_gpu, _args.val_root, _args.resolution, _args.num_class)

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
  _args = parse_args()
  file_path = os.path.join(_args.weight_path, 'model_final.pth')
  model_name = 'bisenetv2_light'

  feature_test = ' float model evaluation'
  if _args.quant_mode > 0:
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
      quant_mode=_args.quant_mode)

  print("-------- End of {} test ".format(model_name))
 
