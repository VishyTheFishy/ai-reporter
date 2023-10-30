"""Stochastic variational inference trainer."""
import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cv2 import imwrite

from BNNBench.bayesian_layers.base import BayesianBase
from BNNBench.backbones.bayesian_unet import define_G
from BNNBench.data.paired_data import get_loader_with_dir

cudnn.benchmark = True


def kl_sum(model):
    kl = 0.0
    for layer in model.modules():
        if isinstance(layer, BayesianBase):
            kl += layer.kl_loss()
    return kl


def train_epoch(
    train_loader,
    model,
    opt,
    init_lr,
    epoch,
    const_lr_epochs,
    total_epochs,
    beta,
):
    for src_img, tgt_img in train_loader:
        src_img, tgt_img = src_img.cuda(), tgt_img.cuda()
        pred = model(src_img)
        l1_loss = F.l1_loss(pred, tgt_img, reduction="mean")
        kl_loss = kl_sum(model)
        loss = l1_loss + beta * kl_loss
        opt.zero_grad()
        loss.backward()
        print(
            f"Epoch {epoch}-of-{total_epochs}, L1 Loss: {l1_loss.item()}, KL loss: {kl_loss.item()}"
        )
        opt.step()
    adjust_learning_rate(opt, init_lr, epoch, const_lr_epochs, total_epochs)


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
        args.src_dir, args.tgt_dir, args.img_size, args.batch_size, True
    )
    # sample one batch to get the number of channels in input and output
    # batch: [B, C, H, W]
    src_batch, tgt_batch = iter(train_loader).__next__()
    in_nc = src_batch.size(1)
    out_nc = tgt_batch.size(1)
    model = define_G(in_nc, out_nc, 64, "unet_256", use_dropout=False)
    opt = optim.Adam(
        model.parameters(),
        args.init_lr,
        betas=(0.5, 0.999),
        weight_decay=args.weight_decay,
    )
    for epoch in range(args.total_epochs):
        train_epoch(
            train_loader,
            model,
            opt,
            args.init_lr,
            epoch,
            args.const_lr_epochs,
            args.total_epochs,
            args.beta,
        )

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
        default=100,
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
    parser.add_argument("--beta", type=float, required=True, help="Beta")
    args = parser.parse_args()
    app(args)
