import random
import os
import warnings
import numpy as np
from PIL import Image
from glob import glob

import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torchvision.transforms import transforms

from pytorch_lightning.core.datamodule import LightningDataModule

from BNNBench.data.paired_data import make_pipeline_fn, AugmentedData, read_image

# https://github.com/xuanqing94/BNNBench/blob/827dabe0a3921d76676481365704b62f19b7a820/BNNBench/data/paired_data.py#L209
PIPELINE_SETTINGS = [
    ("rotate", dict(probability=0.7, max_left_rotation=10, max_right_rotation=10))
]

class NoneTransform(object):
    """ Does nothing to the image, to be used instead of None
    
    Args:
        image in, image out, nothing is done
    """
    def __call__(self, image):       
        return image


class UnalignedDataset(Dataset):

    def __init__(self, dir_A, dir_B, serial_batches=False, transform_A=None, transform_B=None, 
                 max_A_size=-1, max_B_size=-1):

        self.dir_A = dir_A
        self.dir_B = dir_B

        self.A_paths = sorted([os.path.join(self.dir_A, f) for f in os.listdir(self.dir_A)])
        self.B_paths = sorted([os.path.join(self.dir_B, f) for f in os.listdir(self.dir_B)])

        # cap size
        self.A_paths = self.A_paths[:max_A_size]
        self.B_paths = self.B_paths[:max_B_size]
        
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        print(f"A_size={self.A_size}, B_size={self.B_size}")

        self.serial_batches = serial_batches

        self.transform_A = transform_A
        self.transform_B = transform_B

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]

        if self.serial_batches:
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_img = read_image(A_path, autocontrast=False, cutoff=0)[0]
        B_img = read_image(B_path, autocontrast=False, cutoff=0)[0]

        # apply image transformation
        if self.transform_A is not None:
            A = self.transform_A(A_img)
        if self.transform_B is not None:
            B = self.transform_B(B_img)

        return A, B

    def __len__(self):
        return max(self.A_size, self.B_size)

class LitUnalignedDM(LightningDataModule):
    def __init__(self, src_dir, tgt_dir, out_imsize, bsize, num_workers, zoom=None **dset_kwargs):
        super().__init__()
        self.src_dir = src_dir
        self.tgt_dir = tgt_dir
        self.out_imsize = out_imsize
        self.bsize = bsize
        self.num_workers = num_workers
        self.dset_kwargs = dset_kwargs
        self.zoom = zoom
    
    def train_dataloader(self):
        src_trn = os.path.join(self.src_dir, 'train')
        tgt_trn = os.path.join(self.tgt_dir, 'train')

        # https://github.com/xuanqing94/BNNBench/blob/827dabe0a3921d76676481365704b62f19b7a820/BNNBench/data/paired_data.py#L136
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(out_imsize, scale=(zoom, zoom), ratio=(1, 1)) if self.zoom is not None else NoneTransform,
                transforms.RandomCrop(self.out_imsize, padding=12),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(90),
                make_pipeline_fn(PIPELINE_SETTINGS),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        dset_trn = UnalignedDataset(src_trn, tgt_trn, serial_batches=False, 
                                    transform_A=transform, transform_B=transform,
                                    **self.dset_kwargs)
        dl_trn = DataLoader(dset_trn, batch_size=self.bsize, shuffle=True, 
                            drop_last=True, num_workers=self.num_workers)

        return dl_trn

    def test_dataloader(self):
        src_tst = os.path.join(self.src_dir, 'test')
        tgt_tst = os.path.join(self.tgt_dir, 'test')
        # https://github.com/xuanqing94/BNNBench/blob/827dabe0a3921d76676481365704b62f19b7a820/BNNBench/data/paired_data.py#L151
        transform = transforms.Compose(
            [
                transforms.CenterCrop(out_imsize*zoom) if zoom is not None else transforms.CenterCrop(out_imsize),
                transforms.Resize(out_imsize)
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        dset_tst = UnalignedDataset(src_tst, tgt_tst, serial_batches=True, 
                                    transform_A=transform, transform_B=transform,
                                    **self.dset_kwargs)
        dl_tst = DataLoader(dset_tst, batch_size=self.bsize, shuffle=False, 
                            drop_last=False, num_workers=self.num_workers)

        return dl_tst

class LitAlignedDM(LightningDataModule):
    def __init__(self, src_dir, tgt_dir, out_imsize, bsize, num_workers, zoom=None, **kwargs):
        super().__init__()
        self.src_dir = src_dir
        self.tgt_dir = tgt_dir
        self.out_imsize = out_imsize
        self.bsize = bsize
        self.num_workers = num_workers
        self.zoom = zoom
    
    def train_dataloader(self):
        src_trn = os.path.join(self.src_dir, 'train')
        tgt_trn = os.path.join(self.tgt_dir, 'train')

        # https://github.com/xuanqing94/BNNBench/blob/827dabe0a3921d76676481365704b62f19b7a820/BNNBench/data/paired_data.py#L208
        dset_trn = AugmentedData(src_trn, tgt_trn, PIPELINE_SETTINGS, 
                                 self.out_imsize, training=True, zoom=self.zoom)
        sampler_trn = RandomSampler(dset_trn)
        dl_trn = DataLoader(dset_trn, batch_size=self.bsize, sampler=sampler_trn, 
                            num_workers=self.num_workers, pin_memory=True)

        return dl_trn

    def test_dataloader(self):
        src_tst = os.path.join(self.src_dir, 'test')
        tgt_tst = os.path.join(self.tgt_dir, 'test')

        # https://github.com/xuanqing94/BNNBench/blob/827dabe0a3921d76676481365704b62f19b7a820/BNNBench/data/paired_data.py#L208
        dset_tst = AugmentedData(src_tst, tgt_tst, PIPELINE_SETTINGS,
                                 self.out_imsize, training=False, zoom=self.zoom)
        sampler_tst = SequentialSampler(dset_tst)
        dl_tst = DataLoader(dset_tst, batch_size=self.bsize, sampler=sampler_tst, 
                            num_workers=self.num_workers, pin_memory=True)

        return dl_tst

