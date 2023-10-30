"""Ensemble neural networks as an approximate Bayesian learning.

Paper: Uncertainty in Neural Networks: Approximately Bayesian Ensembling, Pearce et al.
Link: https://arxiv.org/pdf/1810.05546.pdf
"""

import argparse

import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from cv2 import imwrite

from BNNBench.backbones.batch_ensemble import batch_unet
from BNNBench.data.paired_data import get_loader_with_dir

cudnn.benchmark = True

def adjust_learning_rate(opt, init_lr, epoch, const_lr_epochs, total_epochs):
    """Decay the learning rate based on schedule"""
    if epoch < const_lr_epochs:
        factor = 1.0
    else:
        factor = 1.0 - (epoch - const_lr_epochs) / (total_epochs - const_lr_epochs + 1)
    lr = init_lr * factor
    for param_group in opt.param_groups:
        param_group["lr"] = lr

def app(args):
    print(str(args))
    train_loader = get_loader_with_dir(
        args.src_dir, args.tgt_dir, args.img_size, args.batch_size, True, drop_last=True,
    )
    # sample one batch to get the number of channels in input and output
    # batch: [B, C, H, W]
    src_batch, tgt_batch = iter(train_loader).__next__()
    in_nc = src_batch.size(1)
    out_nc = tgt_batch.size(1)
    model = batch_unet(args.n_models, in_nc, out_nc, 64, "unet_256", norm="batch", use_dropout=False)
    opt = optim.AdamW(
        model.parameters(),
        args.init_lr,
        betas=(0.5, 0.999),
        weight_decay=args.weight_decay,
    )

    for epoch in range(args.total_epochs):
        for i, (src_img, tgt_img) in enumerate(train_loader):
            src_img, tgt_img = src_img.cuda(), tgt_img.cuda()
            pred = model(src_img)
            l1_loss = F.l1_loss(pred, tgt_img, reduction="mean")
            l1_loss.backward()
            if (i + 1) % 2 == 0:
                print(f"** L1 Loss: {l1_loss.item()}")
                opt.step()
                opt.zero_grad()
        adjust_learning_rate(opt, args.init_lr, epoch, args.const_lr_epochs, args.total_epochs)
    
    import pdb
    pdb.set_trace()
    torch.save(model.state_dict(), args.ckpt_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Launch the ensemble trainer")
    parser.add_argument(
        "--src-dir",
        type=str,
        required=True,
        help="Path to the training source directory",
    )
    parser.add_argument(
        "--tgt-dir",
        type=str,
        required=True,
        help="Path to the training target directory",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument(
        "--img-size", type=int, required=True, help="Size of images in pixels"
    )
    parser.add_argument(
        "--init-lr", type=float, default=2.0e-4, help="Initialized learning rate"
    )
    parser.add_argument(
        "--const-lr-epochs",
        type=int,
        default=50,
        help="Number of epochs at peak learning rate",
    )
    parser.add_argument(
        "--total-epochs", type=int, default=100, help="Total number of epochs"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument(
        "--weight-decay", type=float, default=1.0e-5, help="Weight decay factor"
    )
    parser.add_argument(
        "--ckpt-file", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--n-models", type=int, required=True, help="Number of models to ensemble",
    )
    args = parser.parse_args()
    app(args)
