"""
Folder structure:

/Root(could be train or test)
|
+-- Source image folder
             |
             +-- Image 1
             |
             +-- Image 2
             ...
             +-- Image N
+--- Target image folder
             |
             +-- Image 1
             |
             +-- Image 2
             ...
             +-- Image N             
"""

import os
from itertools import chain
import warnings

import numpy as np
import torch
import torchvision.transforms as transforms
from Augmentor.Pipeline import Pipeline
from imageio import imread, imwrite
from PIL import Image, ImageOps
from tifffile import imread as tif_imread
from tifffile import imwrite as tif_imwrite
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torchvision.transforms.transforms import CenterCrop

class NoneTransform(object):
    """ Does nothing to the image, to be used instead of None
    
    Args:
        image in, image out, nothing is done
    """
    def __call__(self, image):       
        return image



def read_image(img_f, autocontrast: bool, cutoff: float):
    if img_f.endswith(".tif") or img_f.endswith(".tiff"):
        img = tif_imread(img_f)
    else:
        print(img_f)
        img = imread(img_f)

    def fn(arr):
        pil_img = Image.fromarray(arr)
        if autocontrast:
            pil_img = ImageOps.autocontrast(pil_img, cutoff)
        return pil_img

    if len(img.shape) == 2:
        return [fn(img)]
    elif len(img.shape) == 3:
        n_channels = img.shape[-1]
        if n_channels == 1:
            return [fn(img[0, :, :])]
        elif n_channels == 3:
            return [fn(img)]
        else:
            return [fn(img[i, :, :]) for i in range(n_channels)]


def write_image(np_arrs, img_f):
    np_arr = np.concatenate(np_arrs, axis=0)
    ext = img_f.split(".")[-1]
    if ext in ["tif", "tiff"]:
        return tif_imwrite(img_f, np_arr)
    else:
        n_ch = np_arr.shape[0]
        if n_ch not in [1, 3]:
            raise ValueError(
                f"Don't know how to save a {n_ch}-channel image in {ext} format"
            )
        return imwrite(img_f, np_arr)


def read_flat_dir(image_dir: str):
    """Read the images in directory, assuming it is a flat structured folder."""
    result = {}
    for img_f in os.listdir(image_dir):
        # pil_img = read_image(
        #    os.path.join(image_dir, img_f), autocontrast=True, cutoff=0.5
        # )
      pil_img = read_image(
            os.path.join(image_dir, img_f), autocontrast=False, cutoff=0
        )
      result[img_f] = pil_img
    return result


def pair_dirs(src_dir, tgt_dir, **kwargs):
    """Read a pair of directories indicating the source folder and target folder,
    forwarding **kwargs to read_flat_dir().
    """
    src_map = read_flat_dir(src_dir, **kwargs)
    tgt_map = read_flat_dir(tgt_dir, **kwargs)
    if sorted(src_map.keys()) != sorted(
        tgt_map.keys()
    ):
        warnings.warn("Your training folder and test folder have different sets of images")
    common = set(src_map.keys()).intersection(set(tgt_map.keys()))
    if len(common) != len(src_map):
        warnings.warn(f" ===> the actual #images: {len(common)}")
        src_map = {k: v for k, v in src_map.items() if k in common}
        tgt_map = {k: v for k, v in tgt_map.items() if k in common}
    result_pairs = {k: (src_map[k], tgt_map[k]) for k in src_map}
    return result_pairs


def make_pipeline_fn(settings):
    p = Pipeline()
    for aug_name, aug_params in settings:
        getattr(p, aug_name)(**aug_params)
    return p.torch_transform()


def paired_transform(transform_fn):
    def paired_fn(x, y):
        state = torch.get_rng_state().clone()
        x = transform_fn(x)
        # reset random seed to have same random seed
        torch.set_rng_state(state)
        y = transform_fn(y)
        return x, y

    return paired_fn


class AugmentedData(Dataset):
    def __init__(self, src_dir, tgt_dir, settings, out_imsize, training, zoom=None):
        super().__init__()
        self.src_dir = src_dir
        self.tgt_dir = tgt_dir
        if training:              
            self.prep = paired_transform(
                transforms.Compose(
                    [
                        #transforms.RandomResizedCrop(out_imsize, scale=(zoom, zoom), ratio=(1, 1)) if zoom is not None else NoneTransform,
                        transforms.RandomCrop(out_imsize, padding=12),
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.RandomVerticalFlip(0.5),
                        transforms.RandomRotation(90),
                        make_pipeline_fn(settings),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5], std=[0.5]),
                    ]
                ).__call__
            )
        else:
            self.prep = paired_transform(
                transforms.Compose(
                    [
                        transforms.CenterCrop(out_imsize*zoom) if zoom is not None else transforms.CenterCrop(out_imsize),
                        transforms.Resize(out_imsize),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5], std=[0.5]),
                    ]
                ).__call__
            )
        paired_images = []
        image_files = []
        for file_name, (src_imgs, tgt_imgs) in pair_dirs(src_dir, tgt_dir).items():
            paired_images.extend(zip(src_imgs, tgt_imgs))
            image_files.append(file_name)
        self.paired_images = paired_images
        self.image_files = image_files

    def __len__(self):
        return len(self.paired_images)

    def __getitem__(self, i):
        src_img, tgt_img = self.paired_images[i]
        src_img, tgt_img = self.prep(src_img, tgt_img)
        return src_img, tgt_img


def get_loader_with_dir(src_dir, tgt_dir, img_size, batch_size, is_train, drop_last=False):
    if is_train:
        # pipeline_settings = [
        #    ("rotate", dict(probability=0.7, max_left_rotation=10, max_right_rotation=10))
        # ]
        pipeline_settings = []
        dataset = AugmentedData(src_dir, tgt_dir, pipeline_settings, img_size, True)
        # dataset = AugmentedData(src_dir, tgt_dir, pipeline_settings, img_size, False)
        sampler = RandomSampler(dataset)
        loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=drop_last,
        )
        return loader
    else:
        pipeline_settings = []
        dataset = AugmentedData(src_dir, tgt_dir, pipeline_settings, img_size, False)
        sampler = SequentialSampler(dataset)
        loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
        )
        return loader, dataset.image_files


def get_loader(args):
    pipeline_settings = [
        ("rotate", dict(probability=0.7, max_left_rotation=10, max_right_rotation=10))
    ]
    trainset = AugmentedData(
        args.train_src_dir, args.train_tgt_dir, pipeline_settings, args.img_size, True
    )
    testset = AugmentedData(
        args.test_src_dir, args.test_tgt_dir, pipeline_settings, args.img_size, False
    )

    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(
        trainset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = (
        DataLoader(
            testset,
            sampler=test_sampler,
            batch_size=args.test_batch_size,
            num_workers=4,
            pin_memory=True,
        )
        if testset is not None
        else None
    )

    return train_loader, test_loader
