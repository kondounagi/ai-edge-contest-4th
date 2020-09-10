"""Cityscapes Dataloader"""
import os
import random
import numpy as np
import torch
import torch.utils.data as data

from PIL import Image, ImageOps, ImageFilter

__all__ = ['CitySegmentation']


class CitySegmentation(data.Dataset):
    """Cityscapes Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to Cityscapes folder. Default is './datasets/citys'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = CitySegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'signate_datasets'
    NUM_CLASS = 20

    def __init__(self, resize, base_size, crop_size, img_dir='seg_train_images', mask_dir='seg_train_annotations',
                root='./signate_datasets', split='train', mode=None, transform=None):
        super(CitySegmentation, self).__init__()
        self.root = root
        self.split = split
        self.mode = mode if mode is not None else split
        self.transform = transform
        self.base_size = base_size
        self.crop_size = crop_size
        self.resize = resize
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images, self.mask_paths = _get_city_pairs(self.root, self.img_dir, self.mask_dir, self.split) # こいつらはただのパスである
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of: " + self.root + "\n")
        
        self.valid_classes = [29, 64, 69, 70, 75, 76, 82, 93, 115, 117,
                              122, 136, 146, 150, 155, 166, 179, 181, 183, 226]
        self._key = np.ones(228) * -1
        self._key[(np.asarray(self.valid_classes) + 1).tolist()] = range(20)


        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

    def _class_to_index(self, mask): # ok
        values = np.unique(mask)
        for value in values:
            #print("value = ", value)
            assert (value in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def __getitem__(self, index): # ok
        img = Image.open(self.images[index]).convert('RGB') # 256諧調
        mask = Image.open(self.mask_paths[index]).convert('L') # 普通にパスから画像を開いてる

        # resizing
        if img.size == (2048, 1024):
            img = img.resize((self.resize, self.resize // 2))
            mask = mask.resize((self.resize, self.resize // 2), Image.BILINEAR)
        elif img.size == (1936, 1216):
            img = img.resize((self.resize, self.resize * 5 // 8))
            mask = mask.resize((self.resize, self.resize *5 // 8), Image.BILINEAR)
        else :
            raise ValueError
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask

    def _val_sync_transform(self, img, mask): # ok
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask): # ok
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img): # ok
        return np.array(img)

    def _mask_transform(self, mask): # ok
        target = self._class_to_index(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(target).astype('int32'))

    def __len__(self): # ok
        return len(self.images)

    @property
    def num_class(self): # ok
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self): # ok
        return 0


def _get_city_pairs(folder, img_dir, mask_dir, split='train'):
    
    img_paths = os.listdir(os.path.join(folder, img_dir))
    mask_paths = os.listdir(os.path.join(folder, mask_dir))
    for i, filename in enumerate(img_paths):
        img_paths[i] = os.path.join(folder, img_dir, filename)
    for i, filename in enumerate(mask_paths):
        mask_paths[i] = os.path.join(folder, mask_dir, filename)
    img_paths.sort()
    mask_paths.sort()
    
    return img_paths, mask_paths


if __name__ == '__main__':
    dataset = CitySegmentation()
    img, label = dataset[0]
