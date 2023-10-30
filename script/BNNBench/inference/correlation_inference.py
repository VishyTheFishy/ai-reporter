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
    all_model_preds = []
    for model in models:
        all_preds_np = []
        for src_batch, tgt_batch in test_loader:
            src_batch = src_batch.cuda()
            with torch.no_grad():
                pred = model(src_batch)
            pred = pred.cpu().numpy()
            all_preds_np.append(pred.reshape(pred.shape[0], -1))
        all_preds = np.concatenate(all_preds_np, axis=0)
        all_model_preds.append(all_preds)
    all_data = np.stack(all_model_preds, axis=-1)
    def corr(slice):
        return np.corrcoef(slice, rowvar=False)
    print(np.mean([corr(all_data[0]) for i in range(all_data.shape[0])], axis=0))
    # [N_data, D, N_models]
    import pdb
    pdb.set_trace()
    print()

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
