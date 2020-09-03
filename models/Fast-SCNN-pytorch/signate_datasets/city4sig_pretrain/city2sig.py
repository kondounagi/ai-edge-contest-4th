import os
import sys
import shutil
from PIL import Image, ImageOps, ImageFilter
import numpy as np

city_img_folder = '../../datasets/citys/leftImg8bit/train'
city_mask_folder = '../../datasets/citys/gtFine/train'

city2sig_pallette = np.array([0, 70, 0, 0, 0,
                              0, 0, 64, 179, 0,
                              0, 136, 122, 122, 0,
                              0, 0, 146, 0, 226,
                              155, 181, 75, 115, 76,
                              117, 29, 69, 183, 0,
                              0, 82, 117, 150, 0, 0])

for root, _, mask_folder in os.walk(city_mask_folder):
    for mask_path in mask_folder:
        if mask_path.endswith('png') and 'labelIds' in mask_path:
            mask_city = Image.open(os.path.join(root, mask_path))
            new_mask = Image.fromarray(np.uint8(np.maximum(0, city2sig_pallette[np.asarray(mask_city).ravel()]).reshape(1024, 2048))).convert('L')
            new_mask.save(os.path.join('gt', mask_path))


for root, _, img_folder in os.walk(city_img_folder):
    for img_path in img_folder:
        src = os.path.join(root, img_path)
        shutil.copy(src, 'img')