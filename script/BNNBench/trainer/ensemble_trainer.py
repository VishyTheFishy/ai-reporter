"""Ensemble neural networks as an approximate Bayesian learning.

Paper: Uncertainty in Neural Networks: Approximately Bayesian Ensembling, Pearce et al.
Link: https://arxiv.org/pdf/1810.05546.pdf
"""

import argparse

import numpy as np
import scipy.stats as stats
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cv2 import imwrite

from BNNBench.backbones.unet import define_G, init_weights
from BNNBench.data.paired_data import get_loader_with_dir

cudnn.benchmark = True


def make_anchor(model, prior_std):
    """Re-initialize the anchor model"""
    init_weights(model, init_type="normal", init_gain=prior_std)
    for w in model.parameters():
        w.requires_grad_ = False


def dist_to_anchor(model, model_anchor):
    d = 0.0
    n = 0
    for w, w_anchor in zip(model.parameters(), model_anchor.parameters()):
        d += F.mse_loss(w, w_anchor, reduction="sum")
        n += w.numel()
    return d / n

def test_epoch(
    test_loader,
    model,
):
    model.eval()
    corrs = []
    spr = []
    for src_img, tgt_img in test_loader:
        src_img = src_img.cuda()
        with torch.no_grad():
            pred = model(src_img)
        tgt_img = tgt_img.numpy()
        pred = pred.cpu().numpy()
        for i in range(len(pred)):
            c = stats.pearsonr(pred[i].reshape(-1), tgt_img[i].reshape(-1))
            #rho = stats.spearmanr(pred[i].reshape(-1), tgt_img[i].reshape(-1))
            corrs.append(c)
            #spr.append(rho)
    model.train()
    print("===> Testing Corr:", np.mean(corrs))

def train_epoch(
    train_loader,
    model,
    model_anchor,
    opt,
    init_lr,
    epoch,
    const_lr_epochs,
    total_epochs,
    coef,):  
    num_accum = 4
    opt.zero_grad()
    for i, (src_img, tgt_img) in enumerate(train_loader):
        src_img, tgt_img = src_img.cuda(), tgt_img.cuda()
        pred = model(src_img)
        l1_loss = F.l1_loss(pred, tgt_img, reduction="mean")
        anchor_loss = dist_to_anchor(model, model_anchor)
        loss = (l1_loss + coef * anchor_loss)/num_accum
        with open("unet_losses.txt", "w") as loss_file:
            loss_file.write(str(loss.item()))
        loss.backward()
        print(
            f"Epoch {epoch}-of-{total_epochs}, L1 Loss: {l1_loss.item()}, anchor loss: {anchor_loss.item()}"
        )
        if ((i + 1) % num_accum == 0) or (i + 1 == len(train_loader)):
            opt.step()
            opt.zero_grad()

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
    test_loader, _ = get_loader_with_dir(
        args.src_dir, args.tgt_dir, args.img_size, args.batch_size, False
    )
    # sample one batch to get the number of channels in input and output
    # batch: [B, C, H, W]
    src_batch, tgt_batch = iter(train_loader).__next__()
    in_nc = src_batch.size(1)
    out_nc = tgt_batch.size(1)
    model_anchor = define_G(in_nc, out_nc, 64, "unet_256", norm="batch", use_dropout=args.dropout)
    # make_anchor(model_anchor, args.prior_std)
    # model = nn.DataParallel(Unet(in_nc, out_nc, num_down, use_dropout=True).cuda())
    model = define_G(in_nc, out_nc, 64, "unet_256", norm="batch", use_dropout=args.dropout)
    opt = optim.AdamW(
        model.parameters(),
        args.init_lr,
        betas=(0.5, 0.999),
        weight_decay=args.weight_decay,
    )

    for epoch in range(args.total_epochs):
        train_epoch(
            train_loader,
            model,
            model_anchor,
            opt,
            args.init_lr,
            epoch,
            args.const_lr_epochs,
            args.total_epochs,
            args.coef,
        )
        if epoch % 20 == 0:
            test_epoch(test_loader, model)

    torch.save(model.state_dict(), args.ckpt_file)
    # evaluate


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
    parser.add_argument("--dropout", action="store_true", help="If set, will enable dropout")
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
    args = parser.parse_args()
    app(args)
