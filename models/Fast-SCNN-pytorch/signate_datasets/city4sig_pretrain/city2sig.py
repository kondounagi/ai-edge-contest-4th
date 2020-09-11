import os
import sys
import shutil
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import torch
from torchvision import transforms


from tqdm import tqdm

# color pallette

city2sig_pallette = np.array([0, 70, 0, 0, 0,
                              0, 0, 64, 179, 64,
                              0, 136, 122, 122, 0,
                              0, 0, 146, 0, 226,
                              155, 181, 75, 115, 76,
                              117, 29, 69, 183, 0,
                              0, 82, 117, 150, 0, 0])

# transforms

city_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.2869, 0.3252, 0.2839], [0.1756, 0.1805, 0.1772])
])
sig_train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.3744, 0.3841, 0.4029], [0.2778, 0.2752, 0.2596])
])
sig_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.3563, 0.3689, 0.3901], [0.2835, 0.2796, 0.2597])
])
inv_sig_test_transform = transforms.Compose([
    transforms.Normalize([-1.2568, -1.3194, -1.5021], [3.5273, 3.5765, 3.8506])
])

# src_folder以下にあるcityscapes仕様のマスクをすべて、signate仕様のマスクに変換し、dist_folder以下に保存する。
def mask4sig(src_folder, dist_folder):
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
def transform4sig(src_folder, dist_folder, transform):
    for root, _, img_folder in os.walk(src_folder):
        for img_path in tqdm(img_folder):
            src = os.path.join(root, img_path)
            img = Image.open(src)

            img = transform(img)
            img = inv_sig_test_transform(img)

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


def main():
    city_img_folder = 'img1024'
    sig_img_train_folder = '../seg_train_images/1024'
    
    transform4sig(city_img_folder, city_img_folder + '_norm', city_transform)
    transform4sig(sig_img_train_folder, sig_img_train_folder + '_norm', sig_train_transform)
    
    # validate if the mean and std are correct
    calc_mean_std(city_img_folder + '_norm')
    calc_mean_std(sig_img_train_folder + '_norm')

if __name__ == '__main__':
    main()

