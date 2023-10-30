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
import copy

from BNNBench.backbones.unet import define_G, init_weights
from BNNBench.data.paired_data import get_loader_with_dir

cudnn.benchmark = True


def make_anchor(model, prior_std):
    """Re-initialize the anchor model"""
    init_weights(model, init_type="normal", init_gain=prior_std)
    for w in model.parameters():
        w.requires_grad_ = False

def dist_to_anchor(model, model_anchors):
    d = 0.0
    n = 0
    for model_anchor in model_anchors:
        for w, w_anchor in zip(model.parameters(), model_anchor.parameters()):
            d += F.l1_loss(w, w_anchor, reduction="sum")
            n += w.numel()
    return d / n

def train(
    train_loader,
    model,
    model_anchors,
    coef,
    opt,
    scheduler,
    n_iters,
):  
    k_iter = 0
    while k_iter < n_iters:
        for src_img, tgt_img in train_loader:
            src_img, tgt_img = src_img.cuda(), tgt_img.cuda()
            pred = model(src_img)
            l1_loss = F.l1_loss(pred, tgt_img, reduction="mean")
            anchor_loss = dist_to_anchor(model, model_anchors)
            loss = l1_loss - coef * anchor_loss
            opt.zero_grad()
            loss.backward()
            print(
                f"L1 Loss: {l1_loss.item()}, anchor loss: {anchor_loss.item()}"
            )
            opt.step()
            k_iter += 1
            if k_iter == n_iters:
                break
        scheduler.step()


def test(
    test_loader,
    model,
):
    model.eval()
    corrs = []
    for src_img, tgt_img in test_loader:
        src_img = src_img.cuda()
        with torch.no_grad():
            pred = model(src_img)
        tgt_img = tgt_img.numpy()
        pred = pred.cpu().numpy()
        for i in range(len(pred)):
            #c = np.corrcoef(pred[i].reshape(-1), tgt_img[i].reshape(-1))[0, 1]
            c = stats.spearmanr(pred[i].reshape(-1), tgt_img[i].reshape(-1))
            corrs.append(c)
    model.train()
    return np.mean(corrs)

def set_no_grad(model):
    for p in model.parameters():
        p.requires_grad_ = False


def init(args, model_in):
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
    if model_in is None:
        model = define_G(in_nc, out_nc, 64, "unet_256", norm="batch", use_dropout=False)
        model.load_state_dict(torch.load(args.seed_model))
    else:
        model = copy.deepcopy(model_in)
    model_anchor = copy.deepcopy(model)
    #set_no_grad(model_anchor)
    opt = optim.AdamW(
        model.parameters(),
        args.init_lr,
        betas=(0.5, 0.999),
        weight_decay=args.weight_decay,
    )
    def lr_fn(epoch):
        if epoch < 15:
            return 1
        elif epoch < 30:
            return 0.1
        else:
            return 0.1
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    return train_loader, test_loader, model, model_anchor, opt, scheduler


def app(args):
    print(str(args))
    model_in = None
    model_anchors = []
    for model_id in range(6):
        train_loader, test_loader, model, model_anchor, opt, scheduler = init(args, model_in)
        if model_id == 0:
            model_anchors = [model_anchor]
        print("===> Corr:", test(test_loader, model))
        train(train_loader, model, model_anchors, 10, opt, scheduler, 5 * len(train_loader))
        print("===> Corr:", test(test_loader, model))

        print("Finetuning...")
        train(train_loader, model, model_anchors, 0.0, opt, scheduler, 15 * len(train_loader))
        print("===> Corr:", test(test_loader, model))

        torch.save(model.state_dict(), args.ckpt_file + f"_{model_id}")
        model_in = None #model
        model_anchors = [model]

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
        "--seed-model",
        type=str,
        required=True,
        help="Path to the seeding model",
    )
    parser.add_argument(
        "--coef",
        type=float,
        required=True,
        help="The coefficient of losses",
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
        "--total-epochs", type=int, default=200, help="Total number of epochs"
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
