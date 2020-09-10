from .cityscapes import CitySegmentation

datasets = {
    'citys': CitySegmentation,
}


def get_segmentation_dataset(name, resize, base_size, crop_size, img_dir, mask_dir,
                            split='train', mode='train', transform=None):
    """Segmentation Datasets"""
    return datasets[name.lower()](resize, base_size, crop_size, 
                                  img_dir=img_dir, mask_dir=mask_dir,
                                  split=split, mode=mode, transform=transform)
