"""Loading the corrupted CIFAR-10 data."""
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CIFAR_C(Dataset):
    all_corruptions =  [
        "brightness", "defocus_blur", "fog", "gaussian_blur", 
        "glass_blur", "jpeg_compression", "motion_blur", "saturate", 
        "snow", "speckle_noise", "contrast", "elastic_transform",
        "frost", "gaussian_noise", "impulse_noise", "pixelate", 
        "shot_noise", "spatter", "zoom_blur",
    ]
    def __init__(self, data_dir, level, corruptions="all", transform=None) -> None:
        super().__init__()
        data = []
        label = []
        def _load_f(fname):
            return np.load(os.path.join(data_dir, fname))[level*10000:(level+1)*10000]

        if isinstance(corruptions, str) and corruptions == "all":
            # load all files
            for corruption in self.all_corruptions:
                data.append(_load_f(corruption + ".npy"))
            label = [_load_f("labels.npy")] * len(data)
        elif isinstance(corruptions, str):
            data = [_load_f(corruptions + ".npy")]
            label = [_load_f("labels.npy")]
        elif isinstance(corruptions, list):
            data = [_load_f(corruption + ".npy") for corruption in corruptions]
            label = [_load_f("labels.npy")] * len(data)
        else:
            raise RuntimeError("Invalid type of input")
        
        data = np.concatenate(data, axis=0).astype(np.float32) / 255.
        data = np.moveaxis(data, -1, 1)
        self.data = torch.from_numpy(data)
        self.label = torch.from_numpy(np.concatenate(label, axis=0).astype(int))
        self.transform = transform
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.label)

def get_cifar10_c(corruptions, level):
    transform_test = transforms.Compose([
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    loaders = []
    if corruptions == "all":
        for corruption in CIFAR_C.all_corruptions:
            testset = CIFAR_C("./datasets/CIFAR-10-C", level=level, corruptions=corruption, transform=transform_test)
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=1024, shuffle=False, num_workers=2,
            )
            loaders.append(testloader)
    return loaders

def get_cifar100_c(corruptions, level):
    transform_test = transforms.Compose([
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    loaders = []
    if corruptions == "all":
        for corruption in CIFAR_C.all_corruptions:
            testset = CIFAR_C("./datasets/CIFAR-100-C", level=level, corruptions=corruption, transform=transform_test)
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=1024, shuffle=False, num_workers=2,
            )
            loaders.append(testloader)
    return loaders