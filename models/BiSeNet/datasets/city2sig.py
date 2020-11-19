import os
import sys
import shutil
import json
import argparse

import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter

from tqdm import tqdm
from rich.progress import track

np.random.seed(123)

# グレースケールのcolor pallette

city2sig_pallette = np.array([0, 70, 0, 0, 0,
                              0, 0, 64, 179, 64,
                              0, 136, 122, 122, 0,
                              0, 0, 146, 0, 226,
                              155, 181, 75, 115, 76,
                              117, 29, 69, 183, 0,
                              0, 82, 117, 150, 0, 0])

# transformsたち。逆変換が大事なやつ。
# PILのだけど、開くから保存まで一貫してるのでOK
# 順番は(mean, std)
city_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.2869, 0.3252, 0.2839], [0.1756, 0.1805, 0.1772])
])
city_night_aware_transform = transforms.Compose([ # randomに意図的にbrightnessをクソ小さくするので、それを考慮した仮想的なmeanとstd
    transforms.ToTensor(),
    transforms.Normalize([0.2094, 0.2373, 0.2072], [0.1472, 0.1513, 0.1485])
])
sig_train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.3744, 0.3841, 0.4029], [0.2778, 0.2752, 0.2596])
])
sig_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.3563, 0.3689, 0.3901], [0.2835, 0.2796, 0.2597])
])
# こいつ間違ってる。これの逆変換を用意しなくちゃいけない
city_sig_mix_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4117, 0.4309, 0.4600], [0.2756, 0.2635, 0.2277])
])

# inverse transforms
inv_sig_test_transform = transforms.Compose([
    transforms.Normalize([-1.2568, -1.3194, -1.5021], [3.5273, 3.5765, 3.8506])
])
inv_city_mix_transform = transforms.Compose([
    transforms.Normalize([-1.4937, -1.6347, -2.0200], [3.6278, 3.7936, 4.3909])
])
inv_city_night_aware_transform = transforms.Compose([
    transforms.Normalize([-1.4225, -1.5684, -1.3952], [6.7934, 6.6093, 6.7340])
])

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--finetune', action='store_true', default=False)
    parse.add_argument('--pretrain_night', action='store_true', default=False) # 人為的に明るさを時々落とすやつ。
    parse.add_argument('--pretrain_mix', action='store_true', default=False) # signateのよる画像と、cityscapesの人を含む画像
    parse.add_argument('--pretrain', action='store_true', default=False) # 普通のやつ。人を除いただけ

    return parse.parse_args()

# src_folder以下にあるcityscapes仕様のマスクをすべて、signate仕様のマスクに変換し、dist_folder以下に保存する。
def mask2sig(src_folder, dist_folder):
    for root, _, mask_folder in os.walk(src_folder):
        for mask_path in tqdm(mask_folder):
            if mask_path.endswith('png') and 'labelIds' in mask_path:
                mask_city = Image.open(os.path.join(root, mask_path))
                mask_city = city2sig_pallette[np.asarray(mask_city).ravel()]
                mask_city = np.maximum(0, mask_city).reshape(1024, 2048)
                mask_city = np.uint8(mask_city)
                new_mask = Image.fromarray(mask_city).convert('L')
                new_mask.save(os.path.join(dist_folder, mask_path))

# src_folder以下にある画像のstd, meanをsignateの評価環境のものと同じにする。
def img2sig(src_folder, dist_folder, src_transform=city_transform, trg_inv_transform=inv_sig_test_transform):
    for root, _, img_folder in os.walk(src_folder):
        for img_path in tqdm(img_folder):
            src = os.path.join(root, img_path)
            img = Image.open(src)

            img = src_transform(img)
            img = trg_inv_transform(img)

            img = np.transpose(img, (1, 2, 0)) * 255
            
            img = np.minimum(255, img)
            img = np.maximum(0, img)

            img = Image.fromarray(np.uint8(img))
            img.save(os.path.join(dist_folder, img_path))

# src_folderにあるものをすべてダウンサイズする。
def downsize(src_folder, dist_folder, width, height):
    for img_path in tqdm(src_folder):
        img = Image.open(os.path.join(src_folder, img_path))
        img = img.resize((width, height))
        img.save(os.path.join(dist_folder, img_path))

# img_folderに含まれる画像のmeanとstdを計算する。
def calc_mean_std(img_folder):
    imgs = os.listdir(img_folder)
    mean_over_all = 0
    std_over_all = 0
    cnt = 0
    non_transform = transforms.Compose([transforms.ToTensor()])

    for img in tqdm(imgs):
        im = Image.open(os.path.join(img_folder, img)).convert('RGB')
        im = non_transform(im)
        mean = torch.mean(im, axis=1)
        mean = torch.mean(mean, axis=1)
        std = torch.std(im.reshape(3, -1), axis=1)

        mean_over_all += mean
        std_over_all += std
    print("\nmean = ", mean_over_all / len(imgs))
    print("std = ", std_over_all / len(imgs))

#　人も信号も含まれていないマスクを画像とセットで削除。必ず、signate仕様のラベルにされた後に実行されなければならない。
def rm_nohuman_imgs(img_folder, lb_folder):
    lb_paths = os.listdir(lb_folder)
    img_paths = os.listdir(img_folder)
    cnt = 0
    for lb_path in tqdm(lb_paths):
        lb = Image.open(os.path.join(lb_folder, lb_path))
        if (76 not in np.asarray(lb)) and (226 not in np.asarray(lb)): # 人も信号も含まれていないやつ
            os.remove(os.path.join(lb_folder, lb_path))
            os.remove(os.path.join(img_folder, lb_path.replace('gtFine_labelIds', 'leftImg8bit')))
            cnt += 1
    print('removed ', cnt, '　images\n')


# main関数ないは自分の用途に合わせていじってください。　
def main():
    # sources # こっちは基本変えなくていい。
    img_folder_city = 'cityscapes/leftImg8bit'
    lb_folder_city = 'cityscapes/gtFine'

    img_folder_sig = 'signate/seg_train_images'
    lb_folder_sig = 'signate/seg_train_annotations'

    # distinations 変えるならこっちですかね
    img_folder_pretrain = 'pretrain/train/img'
    lb_folder_pretrain = 'pretrain/train/lb'

    img_folder_night = 'pretrain_night/train/img'
    lb_folder_night = 'pretrain_night/train/lb'

    img_folder_mix = 'pretrain_mix/train/img'
    lb_folder_mix = 'pretrain_mix/train/lb'

    img_folder_fine = 'finetune/train/img'
    lb_folder_fine = 'finetune/train/lb'

    img_folder_val_fine = 'finetune/val/img'
    lb_folder_val_fine = 'finetune/val/lb'

    args = parse_args()

    if args.pretrain:
        img2sig(img_folder_city, img_folder_pretrain) 
        mask2sig(lb_folder_city, lb_folder_pretrain)
        # cityscapes のデータの中から、人も信号も映ってないものを削除する。
        rm_nohuman_imgs(img_folder_pretrain, lb_folder_pretrain)

    if args.pretrain_night:
        img2sig(img_folder_city, img_folder_night, src_transform=city_night_aware_transform) 
        mask2sig(lb_folder_city, lb_folder_night)
        # cityscapes のデータの中から、人も信号も映ってないものを削除する。
        rm_nohuman_imgs(img_folder_night, lb_folder_night)

    if args.pretrain_mix:
        img2sig(img_folder_city, img_folder_mix, trg_inv_transform=inv_city_mix_transform) 
        mask2sig(lb_folder_city, lb_folder_mix)
        # cityscapes のデータの中から、人も信号も映ってないものを削除する。
        rm_nohuman_imgs(img_folder_mix, lb_folder_mix)

        # signateのdatasetの中から、夜の画像だけを持ってくる
        jsons_paths = os.listdir(lb_folder_sig)
        for json_filename in track(jsons_paths):
            if json_filename.endswith('json'):
                json_path = os.path.join(lb_folder_sig, json_filename)
                with open(json_path) as json_open:
                    json_data = json.load(json_open)
                if json_data['attributes']['timeofday'] == 'night':
                    img_path = os.path.join(img_folder_sig, json_filename.replace('.json', '.jpg'))
                    lb_path = os.path.join(lb_folder_sig, json_filename.replace('.json', '.png'))
                    shutil.copy(img_path, img_folder_mix)
                    shutil.copy(lb_path, lb_folder_mix)

    # finetuneにデータを移す。これはただ移動させるだけ。
    if args.finetune:
        N = 2243 # signateのデータセットのすべての枚数。
        n_sample = 50 # これはvalに使う枚数。
        val_indices = np.random.choice(N, n_sample) # randomにするのはdatasetの内容が一様分布でないから。特に夜のimageの割合。
        for i in range(N):
            if i in val_indices:
                dist_img = img_folder_val_fine
                dist_lb = lb_folder_val_fine
            else :
                dist_img = img_folder_fine
                dist_lb = lb_folder_fine
            i_str = str(i).zfill(4)

            filename_img = 'train_' + i_str + '.jpg'
            filename_lb = 'train_' + i_str + '.png'

            shutil.copy(os.path.join(img_folder_sig, filename_img), dist_img)
            shutil.copy(os.path.join(lb_folder_sig, filename_lb), dist_lb)
            
if __name__ == '__main__':
    main()

