import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms



def get_tiny_imagenet_loaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=6),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_loader = DataLoader(
        datasets.ImageFolder(root='./datasets/Tiny-ImageNet/train',
            transform=transform_train),
        batch_size=100, shuffle=True, num_workers=2, drop_last=True
    )
    test_loader = DataLoader(
        datasets.ImageFolder(root='./datasets/Tiny-ImageNet/val',
            transform=transform_test),
        batch_size=100, shuffle=False, num_workers=2, drop_last=True
    )
    return train_loader, test_loader
