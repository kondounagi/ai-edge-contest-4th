from .cityscapes import CitySegmentation

datasets = {
    'citys': CitySegmentation,
}


def get_segmentation_dataset(name, resize, base_size, crop_size, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](resize, base_size, crop_size, **kwargs)
