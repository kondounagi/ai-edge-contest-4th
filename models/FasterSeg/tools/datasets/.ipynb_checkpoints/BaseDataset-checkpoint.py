from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np


# preprocess and load
class BaseDataset(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus, split_name):
        super(BaseDataset, self).__init__(batch_size, num_threads, device_id)
        # Cityscapes
#         self.split_name = split_name
#         self.size = np.array([1024, 2048])  # CHW
#         self.downsample = 4
#         self.down_size = (self.size / self.downsample).tolist()   # [384, 768]
#         self.crop_size =  (self.size / self.downsample * 3 / 4).tolist()    # [288, 576] 
#         self.mean = (np.array([0.485, 0.456, 0.406])*255).tolist()
#         self.std = (np.array([0.229, 0.224, 0.225])*255).tolist()
#         img_file_list = '/home/suzuki/ai-edge-contest-dataset/cityscapes_'+split_name+'_images.txt'
#         msk_file_list = '/home/suzuki/ai-edge-contest-dataset/cityscapes_'+split_name+'_annotations.txt'
#         self.img_reader  = ops.FileReader(file_root="//srv/datasets/Cityscapes", file_list=img_file_list, shard_id=device_id, num_shards=num_gpus)
#         self.mask_reader = ops.FileReader(file_root="//srv/datasets/Cityscapes", file_list=msk_file_list, shard_id=device_id, num_shards=num_gpus)
#         self.test_reader = ops.FileReader(file_root="//srv/datasets/Cityscapes", file_list=img_file_list, shard_id=device_id, num_shards=num_gpus)
#         self.img_decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
#         self.mask_decode = ops.ImageDecoder(device = "mixed", output_type = types.GRAY)
#         self.img_resize4train = ops.Resize(device="gpu", interp_type=types.INTERP_LINEAR)
#         self.mask_resize4train = ops.Resize(device="gpu", interp_type=types.INTERP_NN)
#         self.img_resize4val = ops.Resize(device="gpu", interp_type=types.INTERP_LINEAR, size=self.down_size)
#         self.mask_resize4val = ops.Resize(device="gpu", interp_type=types.INTERP_NN, size=self.down_size)
#         self.cmn4image = ops.CropMirrorNormalize(device="gpu", dtype=types.FLOAT, crop=self.crop_size, mean=self.mean, std=self.std)
#         self.cmn4mask = ops.CropMirrorNormalize(device="gpu", dtype=types.FLOAT, crop=self.crop_size, mean=0, std=1)
#         self.norm4image = ops.CropMirrorNormalize(device="gpu", dtype=types.FLOAT, crop=self.down_size, mean=self.mean, std=self.std)
#         self.norm4mask = ops.CropMirrorNormalize(device="gpu", dtype=types.FLOAT, crop=self.down_size, mean=0, std=1)
#         self.rnd = ops.Uniform(range = (0.0, 1.0))
#         self.rsz_rnd = ops.Uniform(range = (0.75, 1.25))
#         self.mirror = ops.CoinFlip(probability = 0.5)
#         self.cast = ops.Cast(device="gpu", dtype=types.DALIDataType.INT64)
        
#         # signate
        self.split_name = split_name
        self.size = np.array([1216, 1936])  # CHW
        self.down_size = self.size * 4 // 8
        self.crop_size = (self.down_size * 3 // 4 // 32 * 32).tolist()
        self.val_size = (self.down_size // 32 * 32).tolist()
        self.mean = (np.array([0.3563, 0.3689, 0.3901])*255).tolist()
        self.std = (np.array([0.2835, 0.2796, 0.259])*255).tolist()
        base_path = '/home/suzuki/ai-edge-contest-dataset/seg_'+split_name
        self.img_reader  = ops.FileReader(file_root=base_path+'_images/dummy', file_list=base_path+'_images.txt' ,shard_id=device_id, num_shards=num_gpus)
        self.mask_reader = ops.FileReader(file_root=base_path+'_annotations_id/dummy', file_list=base_path+'_annotations.txt', shard_id=device_id, num_shards=num_gpus)
        self.img_decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
        self.mask_decode = ops.ImageDecoder(device = "mixed", output_type = types.GRAY)
        self.img_resize4train = ops.Resize(device="gpu", interp_type=types.INTERP_LINEAR)
        self.mask_resize4train = ops.Resize(device="gpu", interp_type=types.INTERP_NN)
        self.img_resize4val = ops.Resize(device="gpu", interp_type=types.INTERP_LINEAR, size=self.val_size)
        self.mask_resize4val = ops.Resize(device="gpu", interp_type=types.INTERP_NN, size=[1216, 1936])
        self.cmn4image = ops.CropMirrorNormalize(device="gpu", dtype=types.FLOAT, crop=self.crop_size, mean=self.mean, std=self.std)
        self.cmn4mask = ops.CropMirrorNormalize(device="gpu", dtype=types.FLOAT, crop=self.crop_size, mean=0, std=1)
        self.norm4image = ops.CropMirrorNormalize(device="gpu", dtype=types.FLOAT, crop=self.val_size, mean=self.mean, std=self.std)
        self.norm4mask = ops.CropMirrorNormalize(device="gpu", dtype=types.FLOAT, crop=[1216, 1936], mean=0, std=1)
        self.rnd = ops.Uniform(range = (0.0, 1.0))
        self.rsz_rnd = ops.Uniform(range = (0.75, 1.25))
        self.mirror = ops.CoinFlip(probability = 0.5)
        self.cast = ops.Cast(device="gpu", dtype=types.DALIDataType.INT64)
        

    def define_graph(self):
        if self.split_name == "train":
            img_files, img_names = self.img_reader(name="img_Reader")
            mask_files, mask_names = self.mask_reader(name="mask_Reader")
            pos_x = self.rnd()
            pos_y = self.rnd()
            rsz_rnd = self.rnd()
            scl = self.rsz_rnd()
            shorter = self.down_size[0] * scl
            is_mirror = self.mirror()
            images = self.img_decode(img_files)
            masks  = self.mask_decode(mask_files)
            images = self.img_resize4train(images, resize_shorter=shorter)
            masks = self.mask_resize4train(masks, resize_shorter=shorter)
            images = self.cmn4image(images, crop_pos_x=pos_x, crop_pos_y=pos_y, mirror=is_mirror)
            masks = self.cmn4mask(masks, crop_pos_x=pos_x, crop_pos_y=pos_y, mirror=is_mirror)
            masks = self.cast(masks)
            assert img_names == mask_names, "img_names={}, mask_names={}".format(img_names, mask_names)
            return [images, masks, img_names]
        
        # imagesはresizeする一方, maskはresizeしない
        elif self.split_name == "val":
            img_files, img_names = self.img_reader(name="img_Reader")
            mask_files, mask_names = self.mask_reader(name="mask_Reader")
            images = self.img_decode(img_files)
            masks  = self.mask_decode(mask_files)
            images = self.img_resize4val(images)
            masks = self.mask_resize4val(masks)
            images = self.norm4image(images)
            masks = self.norm4mask(masks)
            masks = self.cast(masks)
            assert img_names == mask_names, "img_names={}, mask_names={}".format(img_names, mask_names)
            return [images, masks, img_names]
        
        else:
            img_files, names = self.img_reader(name="img_Reader")
            images = self.img_decode(img_files)
            images = self.img_resize4val(images)
            images = self.norm4image(images)
            return [images, names]



# import os
# import cv2
# cv2.setNumThreads(0)
# import torch
# import numpy as np
# from random import shuffle

# import torch.utils.data as data


# class BaseDataset(data.Dataset):
#     def __init__(self, setting, split_name, preprocess=None, file_length=None):
#         super(BaseDataset, self).__init__()
#         self._split_name = split_name
#         self._img_path = setting['img_root']
#         self._gt_path = setting['gt_root']
#         self._portion = setting['portion'] if 'portion' in setting else None
#         self._train_source = setting['train_source']
#         self._eval_source = setting['eval_source']
#         self._test_source = setting['test_source'] if 'test_source' in setting else setting['eval_source']
#         self._down_sampling = setting['down_sampling']
#         print("using downsampling:", self._down_sampling)
#         self._file_names = self._get_file_names(split_name)
#         print("Found %d images"%len(self._file_names))
#         self._file_length = file_length
#         self.preprocess = preprocess

#     def __len__(self):
#         if self._file_length is not None:
#             return self._file_length
#         return len(self._file_names)

#     def __getitem__(self, index):
#         if self._file_length is not None:
#             names = self._construct_new_file_names(self._file_length)[index]
#         else:
#             names = self._file_names[index]
#         img_path = os.path.join(self._img_path, names[0])
#         gt_path = os.path.join(self._gt_path, names[1])
#         item_name = names[1].split("/")[-1].split(".")[0]

#         img, gt = self._fetch_data(img_path, gt_path)
#         img = img[:, :, ::-1]
#         if self.preprocess is not None:
#             img, gt, extra_dict = self.preprocess(img, gt)

#         if self._split_name is 'train':
#             img = torch.from_numpy(np.ascontiguousarray(img)).float()
#             gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
#             if self.preprocess is not None and extra_dict is not None:
#                 for k, v in extra_dict.items():
#                     extra_dict[k] = torch.from_numpy(np.ascontiguousarray(v))
#                     if 'label' in k:
#                         extra_dict[k] = extra_dict[k].long()
#                     if 'img' in k:
#                         extra_dict[k] = extra_dict[k].float()

#         output_dict = dict(data=img, label=gt, fn=str(item_name),
#                            n=len(self._file_names))
#         if self.preprocess is not None and extra_dict is not None:
#             output_dict.update(**extra_dict)

#         return output_dict

#     def _fetch_data(self, img_path, gt_path, dtype=None):
#         img = self._open_image(img_path, down_sampling=self._down_sampling)
#         gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=dtype, down_sampling=self._down_sampling)

#         return img, gt

#     def _get_file_names(self, split_name):
#         assert split_name in ['train', 'val', 'test']
#         source = self._train_source
#         if split_name == "val":
#             source = self._eval_source
#         elif split_name == 'test':
#             source = self._test_source

#         file_names = []
#         with open(source) as f:
#             files = f.readlines()
#         if self._portion is not None:
#             shuffle(files)
#             num_files = len(files)
#             if self._portion > 0:
#                 split = int(np.floor(self._portion * num_files))
#                 files = files[:split]
#             elif self._portion < 0:
#                 split = int(np.floor((1 + self._portion) * num_files))
#                 files = files[split:]

#         for item in files:
#             img_name, gt_name = self._process_item_names(item)
#             file_names.append([img_name, gt_name])

#         return file_names

#     def _construct_new_file_names(self, length):
#         assert isinstance(length, int)
#         files_len = len(self._file_names)
#         new_file_names = self._file_names * (length // files_len)

#         rand_indices = torch.randperm(files_len).tolist()
#         new_indices = rand_indices[:length % files_len]

#         new_file_names += [self._file_names[i] for i in new_indices]

#         return new_file_names

#     @staticmethod
#     def _process_item_names(item):
#         item = item.strip()
#         # item = item.split('\t')
#         item = item.split(' ')
#         img_name = item[0]
#         gt_name = item[1]

#         return img_name, gt_name

#     def get_length(self):
#         return self.__len__()

#     @staticmethod
#     def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None, down_sampling=1):
#         # cv2: B G R
#         # h w c
#         img = np.array(cv2.imread(filepath, mode), dtype=dtype)

#         if isinstance(down_sampling, int):
#             H, W = img.shape[:2]
#             if len(img.shape) == 3:
#                 img = cv2.resize(img, (W // down_sampling, H // down_sampling), interpolation=cv2.INTER_LINEAR)
#             else:
#                 img = cv2.resize(img, (W // down_sampling, H // down_sampling), interpolation=cv2.INTER_NEAREST)
#             assert img.shape[0] == H // down_sampling and img.shape[1] == W // down_sampling
#         else:
#             assert (isinstance(down_sampling, tuple) or isinstance(down_sampling, list)) and len(down_sampling) == 2
#             if len(img.shape) == 3:
#                 img = cv2.resize(img, (down_sampling[1], down_sampling[0]), interpolation=cv2.INTER_LINEAR)
#             else:
#                 img = cv2.resize(img, (down_sampling[1], down_sampling[0]), interpolation=cv2.INTER_NEAREST)
#             assert img.shape[0] == down_sampling[0] and img.shape[1] == down_sampling[1]

#         return img  # cv2の順序

#     @classmethod
#     def get_class_colors(*args):
#         raise NotImplementedError

#     @classmethod
#     def get_class_names(*args):
#         raise NotImplementedError


# if __name__ == "__main__":
#     data_setting = {'img_root': '',
#                     'gt_root': '',
#                     'train_source': '',
#                     'eval_source': ''}
#     bd = BaseDataset(data_setting, 'train', None)
#     print(bd.get_class_names())
