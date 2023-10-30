import pandas as pd
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.transforms import RandomResizedCrop

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from wilds.common.data_loaders import get_eval_loader


def initialize_camelyon17_transform(is_training):
    angles = [0, 90, 180, 270]

    def random_rotation(x: torch.Tensor) -> torch.Tensor:
        angle = angles[torch.randint(low=0, high=len(angles), size=(1,))]
        if angle > 0:
            x = TF.rotate(x, angle)
        return x

    t_random_rotation = transforms.Lambda(lambda x: random_rotation(x))

    if is_training:
        transforms_ls = [
            transforms.Resize(96),
            t_random_rotation,
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    else:
        transforms_ls = [
            transforms.Resize(96),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    transform = transforms.Compose(transforms_ls)
    return transform


def get_camelyon17():
    dataset = get_dataset(
        dataset="camelyon17", version="1.0", root_dir="./datasets/wilds", download=False
    )
    train_tf = initialize_camelyon17_transform(True)
    val_tf = initialize_camelyon17_transform(False)
    train_data = dataset.get_subset("train", frac=1, transform=train_tf)
    train_loader = get_train_loader(
        "standard", train_data, batch_size=36, drop_last=True
    )  # 72
    val_data = dataset.get_subset("val", frac=1, transform=val_tf)
    val_loader = get_eval_loader("standard", val_data, 60)
    test_data = dataset.get_subset("test", frac=1, transform=val_tf)
    test_loader = get_eval_loader("standard", test_data, 60)
    return train_loader, val_loader, test_loader
