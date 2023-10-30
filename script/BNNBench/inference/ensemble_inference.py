import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
from cv2 import imwrite
from torch.nn.functional import dropout

from BNNBench.backbones.unet import define_G
from BNNBench.data.paired_data import get_loader_with_dir


def load_models(args, in_nc, out_nc):
    models = []
    for model_f in args.ckpt_files:
        print(f"Loading {model_f}")
        model = define_G(in_nc, out_nc, 64, "unet_256", norm="batch", use_dropout=False)
        model.load_state_dict(torch.load(model_f))
        model.eval()
        models.append(model)
    return models


def to_png(f, img_arr):
    if img_arr.shape[0] == 1:
        imwrite(f, img_arr.squeeze(0))
    else:
        assert img_arr.shape[0] == 3
        channel_last_img = np.moveaxis(img_arr, 0, -1)
        imwrite(f, channel_last_img)


def app(args):
    print(args)
    test_loader, files = get_loader_with_dir(
        args.test_src_dir, args.test_tgt_dir, args.img_size, args.batch_size, False
    )
    src_batch, tgt_batch = iter(test_loader).__next__()
    in_nc = src_batch.size(1)
    out_nc = tgt_batch.size(1)
    models = load_models(args, in_nc, out_nc)

    image_id = 0
    for src_batch, tgt_batch in test_loader:
        src_batch = src_batch.cuda()
        all_preds_np = []
        all_err_np = []
        for model in models:
            with torch.no_grad():
                pred = model(src_batch)
            pred = pred.cpu()
            err = (pred - tgt_batch).numpy()
            all_err_np.append(err)
            all_preds_np.append(pred.numpy() * 0.5 + 0.5)
        all_preds = np.stack(all_preds_np, axis=0)
        all_err_np = np.stack(all_err_np, axis=0)
        src_img = (
            ((src_batch.cpu().numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        )
        tgt_img = (
            ((tgt_batch.cpu().numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        )
        pred_img = (np.mean(all_preds, axis=0) * 255).clip(0, 255).astype(np.uint8)
        uncertainty = np.std(all_preds, axis=0).astype(float)
        avg_err = np.mean(all_err_np, axis=0).astype(float)
        # save images
        for i in range(uncertainty.shape[0]):
            print(files[image_id])
            out_uncertainty_f = f"{args.result_dir}/uncertainty_{files[image_id]}.npy"
            out_src_f = f"{args.result_dir}/source_{files[image_id]}.png"
            out_tgt_f = f"{args.result_dir}/tgt_{files[image_id]}.png"
            out_pred_f = f"{args.result_dir}/pred_{files[image_id]}.png"
            out_err_f = f"{args.result_dir}/error_{files[image_id]}.npy"

            np.save(out_uncertainty_f, uncertainty[i])
            to_png(out_tgt_f, tgt_img[i])
            to_png(out_src_f, src_img[i])
            to_png(out_pred_f, pred_img[i])
            np.save(out_err_f, avg_err[i])
            image_id += 1
        # import pdb
        # pdb.set_trace()
        # print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference the ensemble Bayesian networks")
    parser.add_argument(
        "--ckpt-files",
        nargs="+",
        type=str,
        required=True,
        help="Checkpoint files in a list",
    )
    parser.add_argument(
        "--img-size", type=int, required=True, help="Size of images in pixel"
    )
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument(
        "--test-src-dir", type=str, help="Path to test source directory"
    )
    parser.add_argument(
        "--test-tgt-dir", type=str, help="Path to test source directory"
    )
    parser.add_argument("--result-dir", type=str, help="Result directory")
    args = parser.parse_args()
    args.ckpt_files = sorted(args.ckpt_files)

    app(args)
