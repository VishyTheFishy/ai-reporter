import numpy as np
import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from BNNBench.backbones.unet import UnetSegmentation, init_weights
from BNNBench.data.paired_data import paired_transform

from torch.utils.data import DataLoader
from torchvision.datasets.cityscapes import Cityscapes
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

#unet = UnetSegmentation(3, 34, num_downs=7, use_dropout=False)
#init_weights(unet)

unet = deeplabv3_resnet50(pretrained=False, num_classes=34).cuda()
#unet = nn.DataParallel(unet).cuda()

crop = paired_transform(transforms.RandomCrop(512, padding=7))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)


def tf(image, target):
    image, target = crop(image, target)
    return normalize(to_tensor(image)), torch.as_tensor(np.array(target),
                                                        dtype=torch.int64)


data = Cityscapes(
    "./datasets/cityscape",
    split="train",
    mode="coarse",
    target_type="semantic",
    transforms=tf,
)

loader = DataLoader(data, batch_size=8, num_workers=4)

loss_f = nn.CrossEntropyLoss()

opt = optim.AdamW(unet.parameters(), lr=1.0e-4)

for epoch in range(200):
    for batch_id, (image, target) in enumerate(loader):
        image, target = image.cuda(), target.cuda()
        pred = unet(image)["out"]
        print(pred.shape, target.shape)
        loss = loss_f(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print("[{}/200][{}/{}] Loss: {}".format(epoch, batch_id, len(loader),
                                                loss.item()))
    torch.save(unet.state_dict(), f"./ckpt/cityscapes/model_{epoch}")
