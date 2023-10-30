import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.cityscapes import Cityscapes

from BNNBench.backbones.unet import UnetSegmentation
from BNNBench.data.paired_data import paired_transform

unet = UnetSegmentation(3, 34, num_downs=7, use_dropout=False)
unet = nn.DataParallel(unet).cuda()
unet.load_state_dict(torch.load("./ckpt/cityscapes/model_199"))
unet.eval()

crop = paired_transform(transforms.CenterCrop(512))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)


def tf(image, target):
    return normalize(to_tensor(image)), \
        torch.as_tensor(np.array(target),
                                                        dtype=torch.int64)


data = Cityscapes(
    "./datasets/cityscape",
    split="val",
    mode="coarse",
    target_type="semantic",
    transforms=tf,
)

loader = DataLoader(data, batch_size=1, num_workers=4)
for batch_id, (image, target) in enumerate(loader):
    image, target = image.cuda(), target.cuda()
    with torch.no_grad():
        pred = unet(image)[:, 3:, :, :]
    idx_max = torch.max(pred, dim=1)[1]
    correct = (idx_max == target).sum()
    prec = correct / target.numel()
    print(prec)
