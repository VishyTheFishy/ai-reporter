"""Ensemble neural networks as an approximate Bayesian learning.

Paper: Uncertainty in Neural Networks: Approximately Bayesian Ensembling, Pearce et al.
Link: https://arxiv.org/pdf/1810.05546.pdf
"""

import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from BNNBench.backbones.unet import define_G, init_weights
from BNNBench.data.paired_data import get_loader_with_dir
from BNNBench.optim.sgld import SGLD

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


def train_epoch(
    train_loader,
    model,
    opt,
    epoch,
    total_epochs,
):
    for src_img, tgt_img in train_loader:
        src_img, tgt_img = src_img.cuda(), tgt_img.cuda()
        pred = model(src_img)
        loss = F.l1_loss(pred, tgt_img, reduction="mean")
        opt.zero_grad()
        loss.backward()
        print(f"Epoch {epoch}-of-{total_epochs}, L1 Loss: {loss.item()}")
        opt.step()

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
    opt = SGLD(
        model.parameters(),
        args.lr,
        num_pseudo_batches=len(train_loader),
        num_burn_in_steps=int(args.total_epochs * len(train_loader)),
    )

    for epoch in range(args.total_epochs):
        train_epoch(
            train_loader,
            model,
            opt,
            epoch,
            args.total_epochs,
        )
        adjust_learning_rate(opt, args.lr, epoch, 40, args.total_epochs)

    state = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
    }
    torch.save(state, args.ckpt_file)
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
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument(
        "--img-size", type=int, required=True, help="Size of images in pixels"
    )
    parser.add_argument(
        "--lr", type=float, default=2.0e-4, help="Initialized learning rate"
    )
    parser.add_argument(
        "--total-epochs",
        type=int,
        default=100,
        help="Number of burn-in epochs",
    )
    parser.add_argument(
        "--ckpt-file", type=str, required=True, help="Path to checkpoint file"
    )
    args = parser.parse_args()
    app(args)
