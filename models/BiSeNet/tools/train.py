#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
sys.path.insert(0, '.')
import os
import os.path as osp
import random
import logging
import time
import argparse
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lib.models import model_factory
from lib.models.bisenetv2 import SegmentHead
from configs import cfg_factory
from lib.signate_cv2 import get_data_loader
from tools.eval import eval_model
from lib.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg
from lib.color_palette import get_palette

from matplotlib import pyplot as plt
import cv2
from PIL import Image
from rich.progress import track

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


# apex
has_apex = True
try:
    from apex import amp, parallel
except ImportError:
    has_apex = False


## fix all random seeds
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic = True

def _matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank', dest='local_rank', type=int, default=-1,)
    parse.add_argument('--model', dest='model', type=str, default='bisenetv2',)
    parse.add_argument('--finetune_from', type=str, default=None,) # 
    parse.add_argument('--resolution', type=int, default=1024) # 横幅（長い方）のこと
    parse.add_argument('--dataset_root', type=str) # fine tune, pre train でデータを簡単に分離できるようにしたい
    parse.add_argument('--val_root', type=str, default='datasets/finetune/val') # fine tune, pre train でデータを簡単に分離できるようにしたい
    parse.add_argument('--num_class', type=int, default=14, choices=[5, 14, 19]) # あんまり引数増やさずに全部変えたいなあ
    parse.add_argument('--lr', type=float, default=5e-2)
    parse.add_argument('--weight_decay', type=float, default=5e-4)
    parse.add_argument('--epochs', type=int, default=1000)
    parse.add_argument('--dataset', type=str, default='cityscapes', choices=['cityscapes', 'signate', 'cityscapes_night'])
    parse.add_argument('--debug', action='store_true', default=False)
    parse.add_argument('--batch-size', type=int, default=32)

    parse.add_argument('--model_type', type=str, default='bisenetv2')
    return parse.parse_args()

args = parse_args()
cfg = cfg_factory[args.model]


def set_model():
    net = model_factory[args.model_type](args.num_class)
    if not args.finetune_from is None:
        net.load_state_dict(torch.load(args.finetune_from))
    
    net.cuda()
    net.train()
    criteria_pre = OhemCELoss(0.7)
    criteria_aux = [OhemCELoss(0.7) for _ in range(cfg.num_aux_heads)]
    return net, criteria_pre, criteria_aux


def set_optimizer(model):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': 0},
            {'params': lr_mul_wd_params, 'lr': cfg.lr_start * 10},
            {'params': lr_mul_nowd_params, 'weight_decay': 0, 'lr': cfg.lr_start * 10},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    optim = torch.optim.SGD(
        params_list,
        lr=cfg.lr_start,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    return optim


def set_meters():
    time_meter = TimeMeter(cfg.max_iter)
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i))
            for i in range(cfg.num_aux_heads)]
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters


def save_model(net, save_pth, is_best=False):
    torch.save(net.state_dict(), save_pth)
    if is_best:
        torch.save(net.state_dict(), osp.join(cfg.respth, 'model_best.pth'))


def train():
    logger = logging.getLogger()
    logger.info(args)
    #is_dist = dist.is_initialized()
    is_dist = False
    
    print("args.model: ", args.model)
    print("args.finetune_from: ", args.finetune_from)
    
    ## dataset
    dl = get_data_loader(
            args.dataset_root, args.resolution, args.num_class,#　使うデータセットはargsから変えられるようにした。
            cfg.ims_per_gpu, cfg.scales, [args.resolution//8, args.resolution//4], # ここにcropsizeがあるので忘れなように。
            cfg.max_iter, mode='train', distributed=is_dist, dataset=args.dataset)

    ## model
    net, criteria_pre, criteria_aux = set_model()

    ## optimizer
    optim = set_optimizer(net)

    ## fp16
    if has_apex:
        opt_level = 'O1' if cfg.use_fp16 else 'O0'
        net, optim = amp.initialize(net, optim, opt_level=opt_level)

    ## meters
    time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters()

    ## lr scheduler
    lr_schdr = WarmupPolyLrScheduler(optim, power=0.9,
        max_iter=cfg.max_iter, warmup_iter=cfg.warmup_iters,
        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)
    # >>> ここがplateau optimizer
    #lr_schdr = ReduceLROnPlateau(optim, mode='max', patience=10, verbose=True)
    print("len(dl)", len(dl))
    miou_best = 0
    n_iter = len(dl)
    ## train loop
    for epoch in range(args.epochs):
        for it, (im, lb) in track(enumerate(dl), description='training', total=n_iter):
            im = im.cuda()
            lb = lb.cuda()

            lb = torch.squeeze(lb, 1)

            optim.zero_grad()
            logits, *logits_aux = net(im)
            loss_pre = criteria_pre(logits, lb)
            loss_aux = [crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)]
            loss = loss_pre + sum(loss_aux)
            writer.add_scalar('loss', loss.item(), n_iter * epoch + it)
            if has_apex:
                with amp.scale_loss(loss, optim) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optim.step()
            #torch.cuda.synchronize() 
            lr_schdr.step()

            time_meter.update()
            loss_meter.update(loss.item())
            loss_pre_meter.update(loss_pre.item())
            _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]

        ## print training log message
        if (epoch + 1) % 25 == 0 or args.debug: # 汚いけどご容赦　
            with torch.no_grad():
                lr = lr_schdr.get_lr()
                lr = sum(lr) / len(lr)
                print_log_msg(
                    epoch, args.epochs, lr, time_meter, loss_meter,
                    loss_pre_meter, loss_aux_meters)
                heads, mious = eval_model(net, 2, args.val_root, args.resolution, args.num_class) # クラス数を柔軟に変えたい 
                miou = np.mean(mious)

                is_best = False
                if miou_best < miou:
                    miou_best = miou
                    is_best = True

                if torch.isnan(loss):
                    raise ValueError('LOSS IS NaN !!!')
                writer.add_scalar('miou', miou, epoch)
                logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))

                ## dump the final model and evaluate the result
                save_pth = osp.join(cfg.respth, 'model_{}.pth'.format(epoch)) # ここもっとスッキリさせようぜ
                logger.info('\nsave models to {}'.format(save_pth))
                save_model(net, save_pth, is_best)

        if (epoch + 1) % 100 == 0 or args.debug: # 推論結果をtensor boardに書き込む。
            with torch.no_grad():
                palette = get_palette(args.num_class)
                dl_eval = get_data_loader(args.val_root, args.resolution, args.num_class, 1, None,
                        None, mode='val', distributed=is_dist)
                net.eval()
                non_transform = transforms.Compose([transforms.ToTensor()])
                for it_eval, (im, lb) in enumerate(dl_eval):
                    im = im.cuda()
                    out = net(im)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
                    pred = palette[out]
                    pred = np.uint8(pred)
                    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
                    pred = non_transform(pred)
                    _matplotlib_imshow(pred) 
                    writer.add_image('val_{}_{}'.format(epoch, it_eval), pred)

    logger.info('\nfinished training')

    return


def main():
    # 仕方ないので、argsでcfgを上書き
    cfg.respth = './logs/res_{}'.format(time.strftime('%Y_%m_%d_%H_%M'))
    cfg.lr_start = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.ims_per_gpu = args.batch_size
    if not osp.exists(cfg.respth): os.makedirs(cfg.respth)
    setup_logger('{}-train'.format(args.model_type), cfg.respth)
    train()
    writer.flush()

if __name__ == "__main__":
    main()