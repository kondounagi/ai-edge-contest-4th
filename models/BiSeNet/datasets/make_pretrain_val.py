import os
import sys
import shutil
import random

import argparse

from tqdm import tqdm


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--val_size', dest='val_size', type=int, default=50)
    parse.add_argument('--reverse', dest='reverse', type=bool, default=False)

    return parse.parse_args()

args = parse_args()


# このコードはcity2sig.pyを実行した後に実行する
def main():
    # sources
    img_folder_pre = 'pretrain/train/img'
    lb_folder_pre = 'pretrain/train/lb'
    
    # destination
    img_folder_val_pre = 'pretrain/val/img'
    lb_folder_val_pre = 'pretrain/val/lb'

    # randomに選んだimgとlbをvalidationセットとして，valに移す
    img_paths_pre = os.listdir(img_folder_pre)
    lb_paths_pre = os.listdir(lb_folder_pre)
    pretrain_size = len(img_paths_pre) # 移動する前のpretrain内のデータ数
    
    img_paths_pre.sort()
    lb_paths_pre.sort()
    '''
    for i in range(pretrain_size):
        print(img_paths_pre[i], lb_paths_pre[i])
    print(pretrain_size)
    '''
    assert len(img_paths_pre) == len(lb_paths_pre)
    
    val_size = args.val_size # valに移す枚数
    assert pretrain_size > val_size

    # valとするpretrainデータののindexをランダムに生成
    val_indices = random.sample(range(0, pretrain_size, 1), k=val_size) 
    print("selected indices for validation: ")
    print(val_indices)
    
    for index in tqdm(val_indices):
        shutil.move(os.path.join(img_folder_pre, img_paths_pre[index]), img_folder_val_pre)
        shutil.move(os.path.join(lb_folder_pre, lb_paths_pre[index]), lb_folder_val_pre)

if __name__ == '__main__':
    main()

