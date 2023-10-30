import argparse


import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from cv2 import imwrite

from BNNBench.backbones.unet import define_G
from BNNBench.data.paired_data import get_loader_with_dir
from BNNBench.optim.sgld import SGLD


def load_model(args, in_nc, out_nc, num_pseudo_batches):
    model = define_G(in_nc, out_nc, 64, "unet_256")
    opt = SGLD(model.parameters(), lr=args.lr, num_pseudo_batches=num_pseudo_batches, num_burn_in_steps=0)
    state = torch.load(args.ckpt_file)
    model.load_state_dict(state['model'])
    return model, opt


def to_png(f, img_arr):
    if img_arr.shape[0] == 1:
        imwrite(f, img_arr.squeeze(0))
    else:
        assert img_arr.shape[0] == 3
        channel_last_img = np.moveaxis(img_arr, 0, -1)
        imwrite(f, channel_last_img)


def train_epoch(
    train_loader,
    model,
    opt,
):
    for src_img, tgt_img in train_loader:
        src_img, tgt_img = src_img.cuda(), tgt_img.cuda()
        pred = model(src_img)
        loss = F.l1_loss(pred, tgt_img, reduction="mean")
        print("Loss: ", loss)
        opt.zero_grad()
        loss.backward()
        opt.step()


def eval_epoch(
    test_loader,
    model,
):
    all_preds = []
    all_err = []
    all_inputs = []
    all_outputs = []
    for src_batch, tgt_batch in test_loader:
        src_batch = src_batch.cuda()
        with torch.no_grad():
            pred = model(src_batch)
        pred = pred.cpu()
        err = pred - tgt_batch
        all_preds.append(pred.numpy())
        all_err.append(err.numpy())
        all_inputs.append(src_batch.cpu().numpy())
        all_outputs.append(tgt_batch.numpy())
    return np.concatenate(all_preds, axis=0), np.concatenate(all_err, axis=0), np.concatenate(all_inputs, axis=0), np.concatenate(all_outputs, axis=0)


def app(args):
    print(args)
    test_loader, files = get_loader_with_dir(
        args.test_src_dir, args.test_tgt_dir, args.img_size, args.batch_size, False
    )
    train_loader = get_loader_with_dir(
        args.train_src_dir, args.train_tgt_dir, args.img_size, args.batch_size, True
    )
    src_batch, tgt_batch = iter(test_loader).__next__()
    in_nc = src_batch.size(1)
    out_nc = tgt_batch.size(1)
    model, opt = load_model(args, in_nc, out_nc, len(train_loader))

    all_pred_samples, all_err_samples = [], []
    for _ in range(args.n_samples):
        # mixin stage
        model.train()
        for _ in range(args.mixing_epochs):
            train_epoch(train_loader, model, opt)
        # sample stage
        model.eval()
        all_preds, all_err, src_img, tgt_img = eval_epoch(test_loader, model)
        all_pred_samples.append(all_preds)
        all_err_samples.append(all_err)
    all_pred_samples = np.stack(all_pred_samples, axis=0)
    uncertainty = np.std(all_pred_samples, axis=0)
    pred_img = ((np.mean(all_pred_samples, axis=0) * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    src_img = ((src_img * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    tgt_img = ((tgt_img * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    for i in range(uncertainty.shape[0]):
        print(files[i])
        out_uncertainty_f = f"{args.result_dir}/uncertainty_{files[i]}.npy"
        out_src_f = f"{args.result_dir}/source_{files[i]}.png"
        out_tgt_f = f"{args.result_dir}/tgt_{files[i]}.png"
        out_pred_f = f"{args.result_dir}/pred_{files[i]}.png"
        # out_err_f = f"{args.result_dir}/error_{files[image_id]}.npy"

        np.save(out_uncertainty_f, uncertainty[i])
        to_png(out_tgt_f, tgt_img[i])
        to_png(out_src_f, src_img[i])
        to_png(out_pred_f, pred_img[i])
        # np.save(out_err_f, avg_err[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference the ensemble Bayesian networks")
    parser.add_argument(
        "--ckpt-file",
        type=str,
        required=True,
        help="Checkpoint file",
    )
    parser.add_argument("--lr", type=float, default=2.0e-4, help="Learning rate")
    parser.add_argument(
        "--img-size", type=int, required=True, help="Size of images in pixel"
    )
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument(
        "--mixing-epochs",
        type=int,
        required=True,
        help="Number of epochs to run before getting next sample",
    )
    parser.add_argument(
        "--n-samples", type=int, required=True, help="Total number of epochs"
    )
    parser.add_argument(
        "--train-src-dir", type=str, help="Path to train source directory"
    )
    parser.add_argument(
        "--train-tgt-dir", type=str, help="Path to train target directory"
    )
    parser.add_argument(
        "--test-src-dir", type=str, help="Path to test source directory"
    )
    parser.add_argument(
        "--test-tgt-dir", type=str, help="Path to test source directory"
    )
    parser.add_argument("--result-dir", type=str, help="Result directory")
    args = parser.parse_args()
    app(args)
