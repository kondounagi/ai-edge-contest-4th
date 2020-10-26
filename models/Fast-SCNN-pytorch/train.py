import os
import argparse
import time
import shutil

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torchvision import transforms
from data_loader import get_segmentation_dataset
from models.fast_scnn import get_fast_scnn
from utils.loss import MixSoftmaxCrossEntropyLoss, MixSoftmaxCrossEntropyOHEMLoss
from utils.lr_scheduler import LRScheduler
from utils.metric import SegmentationMetric

from utils.visualize import get_color_pallete
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import numpy as np
writer = SummaryWriter()


def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Fast-SCNN on PyTorch')
    # model and dataset
    parser.add_argument('--model', type=str, default='fast_scnn',
                        help='model name (default: fast_scnn)')
    parser.add_argument('--dataset', type=str, default='citys',
                        help='dataset name (default: citys)')
    """
    [1536, 1024, 768]
    [768. 512. 384.]
    [576. 384. 288.]
    """
    parser.add_argument('--resize', type=int, default=1536,  # original = 2048
                        help='size of width')                       
    parser.add_argument('--base-size', type=int, default=1536//2, # base_size はresized似合わせる。original = 1024
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=1536*3//8,  # ここも大体比例 original = 768
                        help='crop image size')
    parser.add_argument('--train-split', type=str, default='train',
                        help='dataset train split (default: train)')
    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    parser.add_argument('--epochs', type=int, default=160, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=12,
                        metavar='N', help='input batch size for training (default: 12)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1e-2)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 1e-4)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-folder', default='./weights',
                        help='Directory for saving checkpoint models')
    # evaluation only
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--model_path', type=str,
                        help='use trained model')
    # about dataset for train, val
    parser.add_argument('--sub_out_dir', type=str,
                        help='outdir of pred masks')
    parser.add_argument('--train_img_dir', type=str)
    parser.add_argument('--train_mask_dir', type=str)
    parser.add_argument('--test_img_dir', type=str, default='seg_val_images')
    parser.add_argument('--test_mask_dir', type=str, default='seg_val_annotations')

    # signate specific ?
    parser.add_argument('--use_weight', action='store_true', default=False)
    parser.add_argument('--stage', type=str, default='pre', help='fine or pre')

    # the parser
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    args.device = device
    print(args)
    return args

def _matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.stage = args.stage
        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.3563, 0.3689, 0.3901], [0.2835, 0.2796, 0.2597])
        ])
        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size, 'resize': args.resize}
        train_dataset = get_segmentation_dataset(args.dataset, args.resize, args.base_size, args.crop_size, 
                                                args.train_img_dir, args.train_mask_dir,
                                                split=args.train_split, mode='train', transform=input_transform)
        val_dataset = get_segmentation_dataset(args.dataset, args.resize, args.base_size, args.crop_size, 
                                                args.train_img_dir, args.train_mask_dir,
                                                split='val', mode='testval', transform=input_transform)
        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            drop_last=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_size=1,
                                          shuffle=False)

        # create network
        self.model = get_fast_scnn(args.model_path, dataset=args.dataset, aux=args.aux)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1, 2])
        self.model.to(args.device)

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        # create criterion
        self.criterion = MixSoftmaxCrossEntropyOHEMLoss(aux=args.aux, aux_weight=args.aux_weight,
                                                        use_weight=args.use_weight, ignore_index=-1).to(args.device)

        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        # lr scheduling
        self.lr_scheduler = LRScheduler(mode='poly', base_lr=args.lr, nepochs=args.epochs,
                                        iters_per_epoch=len(self.train_loader), power=0.9)

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)

        self.best_pred = 0.0
    @profile
    def train(self):
        cur_iters = 0
        start_time = time.time()
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.model.train()
            for i, (images, targets) in enumerate(self.train_loader):

                cur_lr = self.lr_scheduler(cur_iters)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = cur_lr
                images = images.to(self.args.device)
                targets = targets.to(self.args.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                cur_iters += 1
                if cur_iters % 10 == 0:
                    print('Epoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.8f || Loss: %.4f' % (
                        epoch, args.epochs, i + 1, len(self.train_loader),
                        time.time() - start_time, cur_lr, loss.item()))
                    writer.add_scalar('loss', loss.item(), cur_iters)

            if self.args.no_val:
                # save every epoch
                save_checkpoint(self.model, self.args, self.stage, is_best=False)
            elif epoch % 20 == 19:
                self.validation(epoch)
            break

        save_checkpoint(self.model, self.args, epoch, self.stage, is_best=False)
        
    def validation(self, epoch):
        is_best = False
        self.metric.reset()
        self.model.eval()
        mIoU_ave = 0
        non_transform = transforms.Compose([transforms.ToTensor()])
        for i, (image, target) in enumerate(self.val_loader):
            image = image.to(self.args.device)

            outputs = self.model(image)
            pred = torch.argmax(outputs[0], 1)
            pred = pred.cpu().data.numpy()
            if 0 < i < 10:
                pred_mask = get_color_pallete(pred.squeeze(0), args.dataset).convert('RGB')
                pred4show = non_transform(pred_mask)
                print(np.asarray(pred_mask).shape)
                _matplotlib_imshow(pred4show)
                writer.add_image('pred_mask_{}_{}'.format(epoch, i), pred4show)
            self.metric.update(pred, target.numpy())
            pixAcc, mIoU = self.metric.get()
            print('Epoch %d, Sample %d, validation pixAcc: %.3f%%, mIoU: %.3f%%' % (
                epoch, i + 1, pixAcc * 100, mIoU * 100))
            mIoU_ave += mIoU
        
        writer.add_scalar('mIoU', mIoU_ave / len(self.val_loader), epoch)

        new_pred = (pixAcc + mIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        save_checkpoint(self.model, self.args, epoch, self.stage, is_best)


def save_checkpoint(model, args, epoch, stage, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_folder)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}_{}_{}.pth'.format(args.model, args.resize, epoch, stage)
    save_path = os.path.join(directory, filename)
    print("saved to ", save_path)
    torch.save(model.state_dict(), save_path)
    if is_best:
        best_filename = '{}_{}_{}_best.pth'.format(args.model, args.resize, stage)
        best_filename = os.path.join(directory, best_filename)
        torch.save(model.state_dict(), best_filename)


if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    if args.eval:
        print('Evaluation model: ', args.resume)
        trainer.validation(args.start_epoch)
    else:
        print('Starting Epoch: %d, Total Epochs: %d' % (args.start_epoch, args.epochs))
        trainer.train()
        writer.flush()

