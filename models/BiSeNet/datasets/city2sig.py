import os
import sys
import shutil
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import torch
from torchvision import transforms


from tqdm import tqdm

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

inv_sig_test_transform = transforms.Compose([
    transforms.Normalize([-1.2568, -1.3194, -1.5021], [3.5273, 3.5765, 3.8506])
])

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
def img2sig(src_folder, dist_folder, src_transform):
    for root, _, img_folder in os.walk(src_folder):
        for img_path in tqdm(img_folder):
            src = os.path.join(root, img_path)
            img = Image.open(src)

            img = src_transform(img)
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


def main():
    # sources # こっちは基本変えなくていい。
    img_folder_city = 'cityscapes/leftImg8bit'
    lb_folder_city = 'cityscapes/gtFine'

    img_folder_sig = 'signate/seg_train_images'
    lb_folder_sig = 'signate/seg_train_annotations'

    # distinations 変えるならこっちですかね
    img_folder_pre = 'pretrain_night/train/img'
    lb_folder_pre = 'pretrain_night/train/lb'

    img_folder_fine = 'finetune/train/img'
    lb_folder_fine = 'finetune/train/lb'

    img_folder_val_fine = 'finetune/val/img'
    lb_folder_val_fine = 'finetune/val/lb'

    # pretrainにまずデータを移す
    img2sig(img_folder_city, img_folder_pre, city_night_aware_transform)
    mask2sig(lb_folder_city, lb_folder_pre)

    # cityscapes のデータの中から、人も信号も映ってないものを削除する。
    #rm_nohuman_imgs(img_folder_pre, lb_folder_pre)

    # finetuneにデータを移す。これはただ移動させるだけ。
    """
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
    """
if __name__ == '__main__':
    main()

