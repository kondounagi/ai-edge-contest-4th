from nvidia.dali.plugin.pytorch import DALIGenericIterator


def get_train_loader(config, Dataset, split_name):
    num_gpus = 1
    if split_name == "train":
        pipes = [Dataset(config["batch_size"], 2, device_id, num_gpus, split_name) for device_id in range(num_gpus)]
    else:
        pipes = [Dataset(1, 2, device_id, num_gpus, split_name) for device_id in range(num_gpus)]
    pipes[0].build()
    
    if split_name in ["train", "val"]:
        dali_iter = DALIGenericIterator(pipelines=pipes, output_map=['data', 'label', 'fn'], reader_name="img_Reader",
                                    auto_reset=True, fill_last_batch=False, dynamic_shape=False)
    else:
        dali_iter = DALIGenericIterator(pipelines=pipes, output_map=['data', 'fn'], reader_name="img_Reader",
                                    auto_reset=True, fill_last_batch=False, dynamic_shape=False)
    
    return dali_iter




# import cv2
# cv2.setNumThreads(0)
# from torch.utils import data

# from utils.img_utils import random_scale, random_mirror, normalize, generate_random_crop_pos, random_crop_pad_to_shape


# class TrainPre(object):
#     def __init__(self, config, img_mean, img_std):
#         self.img_mean = img_mean
#         self.img_std = img_std
#         self.config = config

#     def __call__(self, img, gt):
#         img, gt = random_mirror(img, gt)
#         if self.config.train_scale_array is not None:
#             img, gt, scale = random_scale(img, gt, self.config.train_scale_array)

#         img = normalize(img, self.img_mean, self.img_std)

#         crop_size = (self.config.image_height, self.config.image_width)
#         crop_pos = generate_random_crop_pos(img.shape[:2], crop_size)
#         p_img, _ = random_crop_pad_to_shape(img, crop_pos, crop_size, 0)
#         p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
#         p_gt = cv2.resize(p_gt, (self.config.image_width // self.config.gt_down_sampling, self.config.image_height // self.config.gt_down_sampling), interpolation=cv2.INTER_NEAREST)

#         p_img = p_img.transpose(2, 0, 1)

#         extra_dict = None

#         return p_img, p_gt, extra_dict


# def get_train_loader(config, dataset, portion=None, worker=None, test=False):
#     data_setting = {'img_root': config.img_root_folder,
#                     'gt_root': config.gt_root_folder,
#                     'train_source': config.train_source,
#                     'eval_source': config.eval_source,
#                     'down_sampling': config.down_sampling,
#                     'portion': portion}
#     if test:
#         data_setting = {'img_root': config.img_root_folder,
#                         'gt_root': config.gt_root_folder,
#                         'train_source': config.train_eval_source,
#                         'eval_source': config.eval_source,
#                         'down_sampling': config.down_sampling,
#                         'portion': portion}
#     train_preprocess = TrainPre(config, config.image_mean, config.image_std)

#     train_dataset = dataset(data_setting, "train", train_preprocess, config.batch_size * config.niters_per_epoch)

#     is_shuffle = True
#     batch_size = config.batch_size

#     train_loader = data.DataLoader(train_dataset,
#                                    batch_size=batch_size,
#                                    num_workers=config.num_workers if worker is None else worker,
#                                    drop_last=True,
#                                    shuffle=is_shuffle,
#                                    pin_memory=True)

#     return train_loader
