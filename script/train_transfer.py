
from argparse import ArgumentParser
import numpy as np
import os

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import LitAlignedDM
from model import LitTransferUnet

def parse_arguments():
    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument("--data_dir_A", type=str)
    parser.add_argument("--data_dir_B", type=str)
    parser.add_argument("--bsize", type=int, default=2)
    parser.add_argument("--out_imsize", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every_n_epochs", type=int, default=None)

    parser.add_argument("--max_B_size", type=int, default=-1)
    parser.add_argument("--zoom", type=float, default=None)

    parser.add_argument("--save_dir", type=str, default="./lightning_logs")
    parser.add_argument("--name", type=str, default='')
    parser.add_argument("--version", type=str, default=None)

    parser = LitTransferUnet.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser) # max_epochs

    return parser.parse_args()

def main():
    args = parse_arguments()

    model = LitTransferUnet(**vars(args))

    dm_train = LitAlignedDM(src_dir=os.path.join(args.data_dir_A,'input'), 
                              tgt_dir=os.path.join(args.data_dir_B,'input'), 
                              out_imsize=args.out_imsize, 
                              bsize=args.bsize, 
                              num_workers=args.num_workers,
                              max_B_size=2,
                              zoom=args.zoom)

    dl_train = dm_train.train_dataloader()

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model=model, train_dataloaders=dl_train)

if __name__ == '__main__':
    main()

