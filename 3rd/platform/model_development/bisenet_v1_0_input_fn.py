import os
import cv2


calib_image_dir = r'./calib_images/'
calib_batch_size = 32

def _resize(self, img):
    """ai-edge-contest-4th/models/BiSeNet/lib/base_dataset_signate.pyからもってきた
    cityscapesの場合と、signateの場合でアスペクト比が違うのでそのための場合わけ。
    """
    if img.shape[1] / img.shape[0] == 2:
        img = cv2.resize(img, (self.resolution, self.resolution // 2))
    elif 1 < img.shape[1] / img.shape[0] < 2:
        img = cv2.resize(img, (self.resolution, self.resolution * 5 // 8))
    else:
        raise ValueError('cityscapes or signate dataset only')

    return img


def calib_input(iter):
    images = []
    lines = os.listdir(calib_image_dir)
    for index in range(0, calib_batch_size):
        curline = lines[iter * calib_batch_size + index]
        calib_image_name = curline.strip()
        image = cv2.imread(os.path.join(calib_iamge_dir, calib_image_name))
        image = _resize(image)
        images.append(image)

    return {'input': images}

