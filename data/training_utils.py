import albumentations
import os
import yaml

import data.scale
import numpy as np
import torch

from data.dataset import (
    LSBInstanceDataset,
)


datasets = {
    'instance': {
        'class': LSBInstanceDataset
    },
}


def load_config(config_path, default_config_path=None, default=False):
    if default_config_path is not None:
        config = load_config(default_config_path, default=True)
    else:
        config = {}
    with open(config_path, "r") as stream:
        try:
            config.update(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)

    if not default:
        if 'name' not in config:
            config['name'] = os.path.split(config_path)[-1][:-5]
    return config


if os.path.exists('./dataset_paths.yaml'):
    paths = load_config('./dataset_paths.yaml')
    for dataset_type in datasets.keys():
        if dataset_type in paths:
            datasets[dataset_type].update(paths[dataset_type])


def construct_dataset(dataset='instance', transform=None, idxs=None, bands=['g', 'r'], class_map='basicshells',
                      aug_mult=1, padding=0, images_path=None, ann_path=None):
    Dataset = datasets[dataset]['class']
    if transform is not None:
        transform = get_transform(transform)
    if ann_path is None:
        if 'annotations' in datasets[dataset]:
            ann_path = datasets[dataset]['annotations']
        else:
            raise ValueError('annotations path not passed and it is not set in ./dataset_paths.yaml')
    if images_path is None:
        if 'images' in datasets[dataset]:
            images_path = datasets[dataset]['images']
        else:
            raise ValueError('images path not passed and it is not set in ./dataset_paths.yaml')
    return Dataset(
        images_path,
        ann_path,
        bands=bands,
        class_map=class_map,
        indices=idxs,
        aug_mult=aug_mult,
        transform=transform,
        padding=padding,
    )


def get_transform(transforms):
    def parse_args(args):
        pos_args = []
        kwargs = {}
        if type(args) is list:
            pos_args += args
        if type(args) is dict:
            kwargs.update(args)
        return pos_args, kwargs
    transforms = [TRANSFORMS[t](*parse_args(args)[0], **parse_args(args)[1]) for t, args in transforms.items()]
    return albumentations.Compose(transforms)


# Some gross patching to prevent albumentations clipping image
def gauss_apply(self, img, gauss=None, **params):
    img = img.astype("float32")
    # print((img+gauss).min(), (img+gauss).max(), (img).min(), (img).max())
    return img + gauss


# More gross patching to force albumentations to take entire image if cropsize > imagesize
def random_crop(img: np.ndarray, crop_height: int, crop_width: int, h_start: float, w_start: float):
    height, width = img.shape[:2]
    if height < crop_height:
        crop_height = height
    if width < crop_width:
        crop_width = width
    x1, y1, x2, y2 = albumentations.augmentations.crops.functional.get_random_crop_coords(
        height, width, crop_height, crop_width, h_start, w_start)
    img = img[y1:y2, x1:x2]
    return img


def crop_apply(self, img, h_start=0, w_start=0, **params):
    return random_crop(img, self.height, self.width, h_start, w_start)


def contrast_apply(self, img, alpha=1.0, beta=0.0, **params):
    img = img.astype("float32")

    if alpha != 1:
        img *= alpha
    if beta != 0:
        if self.beta_by_max:
            max_value = 1.
            img += beta * max_value
        else:
            img += beta * np.mean(img)
    return img


albumentations.RandomContrast.apply = contrast_apply
albumentations.GaussNoise.apply = gauss_apply
albumentations.RandomCrop.apply = crop_apply
albumentations.CenterCrop.apply = crop_apply

TRANSFORMS = {
    'crop': albumentations.RandomCrop,
    'resize': albumentations.Resize,
    'pad': albumentations.PadIfNeeded,
    'flip': albumentations.Flip,
    'rotate': albumentations.RandomRotate90,
    'noise': albumentations.GaussNoise,
    'affine': albumentations.Affine,
    'contrast': albumentations.RandomContrast,
    'center_crop': albumentations.CenterCrop,
}


def lsb_datasets(class_map, dataset='instance'):
    # split the dataset in train and test set
    dataset = construct_dataset(dataset=dataset, class_map=class_map)
    N = len(dataset)

    # define proportion of dataset where to start test&validation sets
    # currently validation set has size zero, and test is 15% of dataset
    test_p = .85
    val_p = .85
    indices = torch.randperm(int(N * test_p)).tolist()
    test_indices = torch.arange(int(N * test_p), N).tolist()

    # define transform
    transform = {
        'resize': [1024, 1024],
        'flip': None,
        'rotate': None,
        'noise': {'var_limit': .1, 'p': .8},
        'contrast': {'limit': 0.02}
    }

    # get datasets
    dataset_train = construct_dataset(
        idxs=indices[:int(N * val_p)],
        class_map=class_map,
        transform=transform,
        aug_mult=4)
    dataset_val = construct_dataset(
        idxs=indices[int(N * val_p):int(N * test_p)],
        class_map=class_map,
        transform=transform)
    dataset_test = construct_dataset(
        idxs=test_indices,
        class_map=class_map,
        transform={'resize': [1024, 1024]})

    return dataset_train, dataset_val, dataset_test


def get_scale(scale_key, n_channels):
    if scale_key is not None:
        scale = data.scale.get_scale(scale_key)(n_channels)
        n_scaling = scale.n_scaling
    else:
        scale = None
        n_scaling = 1
    return scale, n_scaling
