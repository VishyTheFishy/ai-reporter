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

from BNNBench.backbones.unet import define_G, init_weights
from BNNBench.data.paired_data import get_loader_with_dir

cudnn.benchmark = True


def adjust_learning_rate(n_iter, total_iters, n_ensemble, on_stage_finished):
    """Decay the learning rate based on schedule"""
    iter_per_model = math.ceil(total_iters / n_ensemble)
    phase = ((n_iter - 1) % iter_per_model) / iter_per_model
    print(phase)
    if phase == 0:
        on_stage_finished(n_iter // iter_per_model)
    return 0.5 * (math.cos(math.pi * phase) + 1.0)


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
    model = define_G(in_nc, out_nc, 64, "unet_256", norm="batch", use_dropout=False)
    def save(model_id):
        torch.save(model.state_dict(), args.ckpt_file + f"_{model_id}")

    opt = optim.AdamW(
        model.parameters(),
        args.init_lr,
        betas=(0.5, 0.999),
        weight_decay=args.weight_decay,
    )
    lr_scheduler = LambdaLR(opt, lambda n_iter: adjust_learning_rate(n_iter, args.total_epochs * len(train_loader), args.n_models, save))

    for epoch in range(args.total_epochs):
        for src_img, tgt_img in train_loader:
            src_img, tgt_img = src_img.cuda(), tgt_img.cuda()
            pred = model(src_img)
            l1_loss = F.l1_loss(pred, tgt_img, reduction="mean")
            opt.zero_grad()
            l1_loss.backward()
            print(f"** L1 Loss: {l1_loss.item()}")
            opt.step()
            lr_scheduler.step()

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
    parser.add_argument(
        "--prior-std",
        type=float,
        default=0.1,
        help="Standard deviation in anchor network",
    )
    parser.add_argument(
        "--coef",
        type=float,
        default=0,
        help="Coefficient between regression loss and anchor loss",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument(
        "--img-size", type=int, required=True, help="Size of images in pixels"
    )
    parser.add_argument(
        "--init-lr", type=float, default=2.0e-4, help="Initialized learning rate"
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
