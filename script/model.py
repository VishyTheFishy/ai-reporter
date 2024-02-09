from argparse import ArgumentParser
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

import torchmetrics
import pytorch_lightning as pl

# from BNNBench.backbones.unet import define_G
from networks import define_G, define_D
from utils import get_constant_dim_mask
from msu_net import MSU_Net

class LitI2IPaired(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitI2IPaired")
        parser.add_argument("--pretrained_unet_path", type=str)

        # default train config pulled from pix2pix repo
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/14422fb8486a4a2bd991082c1cda50c3a41a755e/options/base_options.py#L31
        parser.add_argument("--in_nc", type=int, default=3)
        parser.add_argument("--out_nc", type=int, default=3)
        parser.add_argument("--ngf", type=int, default=64)
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/f13aab8148bd5f15b9eb47b690496df8dadbab0c/options/train_options.py#L30
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--b1", type=float, default=0.5)
        parser.add_argument("--b2", type=float, default=0.999)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        parser.add_argument('--n_epochs_const', type=int, default=50, 
                            help='number of epochs with the initial learning rate')
        # parser.add_argument("--step_freq_D", type=int, default=1)
        parser.add_argument("--no_dropout", action='store_true')

        return parent_parser

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self._init_models()
        self._init_metrics()

    def _init_models(self):
        raise NotImplementedError

    def _init_metrics(self):
        self.l1_loss = nn.L1Loss()
        self.pearson_val = []
        self.pearson_tst = []
        self.pearson_metric = torchmetrics.PearsonCorrCoef()

    def training_step(self, batch):
        raise NotImplementedError

    def validation_step(self, batch):
        raise NotImplementedError

    def test_step(self, batch):
        raise NotImplementedError

    def on_validation_epoch_end(self, *args, **kwargs):
        self.log('pearson_val', torch.tensor(self.pearson_val).mean(), sync_dist=True)
        self.pearson_val = [] # reset

    def on_test_epoch_end(self, *args, **kwargs):
        #self.log('pearson_tst', np.mean(self.pearson_tst))
        self.log('pearson_tst', torch.tensor(self.pearson_tst).mean(), sync_dist=True)
        self.pearson_tst = [] # reset

    def configure_optimizers(self):
        opt = optim.AdamW(self.model.parameters(), lr=self.hparams.lr, 
                          betas=(self.hparams.b1, self.hparams.b2),
                          weight_decay=self.hparams.weight_decay)

        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/f13aab8148bd5f15b9eb47b690496df8dadbab0c/models/networks.py#L46
        def lambda_rule(epoch):
            n_epochs_decay = float(self.hparams.max_epochs - self.hparams.n_epochs_const + 1)
            lr_l = 1.0 - max(0, epoch - self.hparams.n_epochs_const) / n_epochs_decay
            return lr_l

        sch = lr_scheduler.LambdaLR(opt, lr_lambda=lambda_rule)

        return [opt], [sch]

class LitMSUnet(LitI2IPaired):
    def _init_models(self):
        self.model = MSU_Net(self.hparams.in_nc, self.hparams.out_nc)
        if (self.hparams.pretrained_unet_path != 'None') and \
           (self.hparams.pretrained_unet_path is not None):
            self.model.load_state_dict(torch.load(self.hparams.pretrained_unet_path))
            print(f"Loading pretrained weight from {self.hparams.pretrained_unet_path}")

    def training_step(self, batch):
        src, tgt = batch
        tgt_pred = self.model(src)
        loss = self.l1_loss(tgt_pred, tgt)
        return loss

    def _eval_pearson(self, batch):
        src, tgt = batch

        mask = get_constant_dim_mask(tgt[0].detach().cpu().numpy())
        mask = torch.from_numpy(mask).to(src.device)

        with torch.no_grad():
            tgt_pred = self.model(src)[:, mask, :, :]
            tgt = tgt[:, mask, :, :]
            p = self.pearson_metric(tgt_pred.flatten(), tgt.flatten())
        return p

    def validation_step(self, batch, batch_idx):
        self.pearson_val.append(self._eval_pearson(batch))

    def test_step(self, batch, batch_idx):
        self.pearson_tst.append(self._eval_pearson(batch))

class LitMSUnetV2(LitMSUnet):
    def _init_models(self):
        self.model = define_G(self.hparams.in_nc, self.hparams.out_nc, 
                              self.hparams.ngf, "msunet_256", norm="batch", 
                              use_dropout=not self.hparams.no_dropout)
        if (self.hparams.pretrained_unet_path != 'None') and \
           (self.hparams.pretrained_unet_path is not None):
            self.model.load_state_dict(torch.load(self.hparams.pretrained_unet_path))
            print(f"Loading pretrained weight from {self.hparams.pretrained_unet_path}")

class LitI2IGAN(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitI2IGAN")
        parser.add_argument("--pretrained_unet_path", type=str)

        # default train config pulled from pix2pix repo
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/14422fb8486a4a2bd991082c1cda50c3a41a755e/options/base_options.py#L31
        parser.add_argument("--in_nc", type=int, default=3)
        parser.add_argument("--out_nc", type=int, default=3)
        parser.add_argument("--ngf", type=int, default=64)
        parser.add_argument("--ndf", type=int, default=64)
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/f13aab8148bd5f15b9eb47b690496df8dadbab0c/options/train_options.py#L30
        parser.add_argument("--lr_G", type=float, default=2e-4)
        parser.add_argument("--lr_D", type=float, default=2e-4)
        parser.add_argument("--b1_G", type=float, default=0.5)
        parser.add_argument("--b1_D", type=float, default=0.5)
        parser.add_argument("--b2_G", type=float, default=0.999)
        parser.add_argument("--b2_D", type=float, default=0.999)
        parser.add_argument("--weight_decay_G", type=float, default=1e-5)
        parser.add_argument("--weight_decay_D", type=float, default=1e-5)
        parser.add_argument('--n_epochs_const', type=int, default=100, 
                            help='number of epochs with the initial learning rate')
        # parser.add_argument("--step_freq_D", type=int, default=1)
        parser.add_argument("--no_dropout_G", action='store_true')
        parser.add_argument("--generator_advantage", type=int, default=1)
        parser.add_argument("--adaptation_layer", type=int, default=-1)
        parser.add_argument("--weight_id", type=int, default=0)


        return parent_parser

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self._init_models()
        self._init_metrics()

    def _init_models(self):
        raise NotImplementedError

    def _init_metrics(self):
        self.bce_logits = nn.BCEWithLogitsLoss()
        self.pearson_val = []
        self.pearson_tst = []
        self.pearson_metric = torchmetrics.PearsonCorrCoef()

    def training_step(self, batch, batch_idx, optimizer_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_validation_epoch_end(self, *args, **kwargs):
        self.log('pearson_val', torch.tensor(self.pearson_val).mean())
        self.pearson_val = [] # reset

    def on_test_epoch_end(self, *args, **kwargs):
        #self.log('pearson_tst', np.mean(self.pearson_tst))
        self.log('pearson_tst', torch.tensor(self.pearson_tst).mean())
        self.pearson_tst = [] # reset

    def configure_optimizers(self):
        opt_G = optim.AdamW(self.G.parameters(), lr=self.hparams.lr_G, 
                            betas=(self.hparams.b1_G, self.hparams.b2_G),
                            weight_decay=self.hparams.weight_decay_G)
        opt_D = optim.AdamW(self.D.parameters(), lr=self.hparams.lr_D, 
                            betas=(self.hparams.b1_D, self.hparams.b2_D),
                            weight_decay=self.hparams.weight_decay_D)

        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/f13aab8148bd5f15b9eb47b690496df8dadbab0c/models/networks.py#L46
        def lambda_rule(epoch):
            n_epochs_decay = float(100 - self.hparams.n_epochs_const + 1) #self.hparams.max_epochs
            lr_l = 1.0 - max(0, epoch - self.hparams.n_epochs_const) / n_epochs_decay
            return lr_l

        sch_G = lr_scheduler.LambdaLR(opt_G, lr_lambda=lambda_rule)
        sch_D = lr_scheduler.LambdaLR(opt_D, lr_lambda=lambda_rule)

        '''
        optimizers = [{'optimizer': opt_D, 'frequency': self.hparams.step_freq_D, 'lr_scheduler': sch_D},
                      {'optimizer': opt_G, 'frequency': 1, 'lr_scheduler': sch_G}]
        '''
        opts = [opt_D, opt_G]
        schs = [sch_D, sch_G]

        return opts, schs

class LitUnetGAN(LitI2IGAN):

    def _init_models(self):
        self.G = define_G(self.hparams.in_nc, self.hparams.out_nc, 
                          self.hparams.ngf, "unet_256", norm="batch", 
                          use_dropout=not self.hparams.no_dropout_G)
        if (self.hparams.pretrained_unet_path != 'None') and \
           (self.hparams.pretrained_unet_path is not None):
            state_dict = torch.load(self.hparams.pretrained_unet_path)
            old_dict = state_dict
            state_dict = {}
            for key, value in old_dict.items():
                new_key = key.replace('module.', '')  # Remove "module." from the key
                state_dict[new_key] = value
            self.G.load_state_dict(state_dict)
            print(f"Loading pretrained weight from {self.hparams.pretrained_unet_path}")

        self.D = define_D(self.hparams.out_nc, self.hparams.ndf, 'basic',
                          n_layers_D=3, norm="batch")

    def training_step(self, batch, batch_idx, optimizer_idx):
        src, tgt_real = batch
        tgt_fake = self.G(src)
        # D
        if optimizer_idx == 0:
            pred_y = self.D(tgt_real)
            y_real = torch.ones_like(pred_y)
            loss_A = self.bce_logits(pred_y, y_real)

            pred_y = self.D(tgt_fake.detach())
            y_fake = torch.zeros_like(pred_y)
            loss_B = self.bce_logits(pred_y, y_fake)

            loss_d = (loss_A + loss_B) / 2
            self.log("loss_d", loss_d, prog_bar=True, logger=True)
            return loss_d
        # G
        elif optimizer_idx == 1:
            pred_y = self.D(tgt_fake)
            y_fake = torch.ones_like(pred_y, requires_grad=False)
            loss_g = self.bce_logits(pred_y, y_fake)
            self.log("loss_g", loss_g, prog_bar=True, logger=True)
            return loss_g
        else:
            raise NotImplementedError

    def _eval_pearson(self, batch):
        src, tgt_real = batch

        mask = get_constant_dim_mask(tgt_real[0].detach().cpu().numpy())
        mask = torch.from_numpy(mask).to(src.device)

        with torch.no_grad():
            tgt_fake = self.G(src)[:, mask, :, :]
            tgt_real = tgt_real[:, mask, :, :]
            p = self.pearson_metric(tgt_fake.flatten(), tgt_real.flatten())
        return p

    def validation_step(self, batch, batch_idx):
        self.pearson_val.append(self._eval_pearson(batch))

    def test_step(self, batch, batch_idx):
        self.pearson_tst.append(self._eval_pearson(batch))

class LitMSUnetGAN(LitUnetGAN):
    def _init_models(self):
        self.G = MSU_Net(self.hparams.in_nc, self.hparams.out_nc)
        if (self.hparams.pretrained_unet_path != 'None') and \
           (self.hparams.pretrained_unet_path is not None):
            self.G.load_state_dict(torch.load(self.hparams.pretrained_unet_path))
            print(f"Loading pretrained weight from {self.hparams.pretrained_unet_path}")

        self.D = define_D(self.hparams.out_nc, self.hparams.ndf, 'basic',
                          n_layers_D=3, norm="batch")

class LitAddaUnet(LitI2IGAN):

    def _init_models(self):
        channels_dict = [64,128, 256,512, 512,512, 512, 512,512,512, 512, 512, 256, 128, 64, 3]
        kw_dict = [4,4, 4,4, 4,3, 3,3, 3,3, 4,4, 4,4, 4,4]
        old_dict = torch.load(self.hparams.pretrained_unet_path)
        state_dict = {}
        for key, value in old_dict.items():
            new_key = key.replace('module.', '')  # Remove "module." from the key
            state_dict[new_key] = value

        self.G_A = define_G(self.hparams.in_nc, self.hparams.out_nc, 
                            self.hparams.ngf, "unet_256", norm="batch", 
                            use_dropout=not self.hparams.no_dropout_G).eval()
        self.G_A.load_state_dict(state_dict)

        for p in self.G_A.parameters():
            p.requires_grad = False

        self.G = define_G(self.hparams.in_nc, self.hparams.out_nc, 
                          self.hparams.ngf, "unet_256", norm="batch", 
                          use_dropout=not self.hparams.no_dropout_G)
        self.G.load_state_dict(state_dict)

        self.D_list = nn.ModuleList([define_D(channels, self.hparams.ndf, 'basic', n_layers_D=3, norm="batch", kw=kw).to("cuda") for channels, kw in zip(channels_dict, kw_dict)])
        self.D_losses = [[] for _ in range(len(self.D_list))]
        self.D = self.D_list[self.hparams.adaptation_layer]

    def configure_optimizers(self):
        gen_optimizer = optim.Adam(self.G.parameters(), lr=self.hparams.lr_G)
        disc_optimizers = [optim.Adam(D.parameters(), lr=self.hparams.lr_D) for D in self.D_list]
        return [gen_optimizer] + disc_optimizers

    def training_step(self, batch, batch_idx, optimizer_idx):
        src_A, src_B = batch
        # D
        if optimizer_idx > 0:
            with torch.no_grad():
                tgt_A = self.G_A(src_A, layer_n=(optimizer_idx-1))
                tgt_B = self.G(src_B, layer_n=(optimizer_idx-1))

            pred_y = self.D_list[(optimizer_idx-1)](tgt_A)
            y_A = torch.ones_like(pred_y)
            loss_A = self.bce_logits(pred_y, y_A)
    
            pred_y = self.D_list[(optimizer_idx-1)](tgt_B)
            y_B = torch.zeros_like(pred_y)
            loss_B = self.bce_logits(pred_y, y_B)
    
            loss_d = (loss_A + loss_B) / 2
            self.log(f"loss_d:{optimizer_idx-1}", loss_d, prog_bar=True, logger=True)
            self.D_losses[(optimizer_idx-1)].append(loss_d.item())
            return loss_d

        # G
        elif optimizer_idx == 0:
            layers = (0,1,2,3,4,5,9,10,11,12,13,14,15)
            weights_list = [[0.07747483149863384 ,0.0783129911313119 ,0.07831298963085628 ,0.07831297743720429 ,0.0293811130126472 ,8.221884096058708e-48 ,0.07831195880681957 ,4.422896237663228e-33 ,0.031712807498803536 ,0.07830244609273487 ,0.0783129911294682 ,0.078312991131312 ,0.078312991131312 ,0.07831299105991217 ,0.07831299113008058 ,0.07831292930890345 ,],
                           [0.007628435055044103 ,0.11118050889110241 ,0.09256125726869514 ,0.08326877561750291 ,0.00014884420004778625 ,5.1538816451499345e-09 ,0.05544475942167392 ,1.8093979138111243e-08 ,0.0001655805111525593 ,0.036584451149387166 ,0.10651626788867803 ,0.11163476816735646 ,0.11195407099742753 ,0.10095096380180218 ,0.10692842248448746 ,0.07503287129778165 ,],
                            [4.6961623089611685e-05 ,0.18874584701244707 ,0.028732142936136903 ,0.016155359619449097 ,3.211113312586784e-07 ,8.841386766024644e-11 ,0.0033630477292582468 ,2.0289909936487102e-10 ,3.5876733133113703e-07 ,0.0010439219441199278 ,0.08950237694730846 ,0.22129654694312825 ,0.29429847239147083 ,0.052921038028376687 ,0.09381826554897117 ,0.010075339106268288 ,],
                        [2.0092680530827732e-10 ,0.006397115597401408 ,9.830414474095416e-06 ,2.5611929846086835e-06 ,7.106751256813947e-13 ,4.467763266363773e-16 ,1.1534230841763144e-07 ,8.779402640264767e-16 ,7.957732115214074e-13 ,1.6160087617603004e-08 ,0.0002601607089838945 ,0.017384514479146435 ,0.9755881680770355 ,4.992134458146516e-05 ,0.0003066598068566724 ,9.366737052715126e-07 ,],]
            weight = weights_list[weight_id]
            
            loss_g = 0
            for layer in layers:
                with torch.no_grad():
                    tgt_A = self.G_A(src_A, layer_n=layer)
                tgt_B = self.G(src_B, layer_n=layer)
                pred_y = self.D_list[layer](tgt_B)
                y_A = torch.ones_like(pred_y, requires_grad=False)
                loss_g_l = self.bce_logits(pred_y, y_A)*weight[layer]
                print(layer, loss_g_l)
                loss_g += loss_g_l
            self.log("loss_g", loss_g, prog_bar=True, logger=True)
            return loss_g

        else:
            raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        src_B, tgt_B = batch

        mask = get_constant_dim_mask(tgt_B[0].detach().cpu().numpy())
        mask = torch.from_numpy(mask).to(src_B.device)

        with torch.no_grad():
            pred_tgt_B = self.G(src_B)[:, mask, :, :]
            tgt_B = tgt_B[:, mask, :, :]
            p = self.pearson_metric(pred_tgt_B.flatten(), tgt_B.flatten())            
            self.pearson_val.append(p)

    def test_step(self, batch, batch_idx):
        src_B, tgt_B = batch

        mask = get_constant_dim_mask(tgt_B[0].detach().cpu().numpy())
        mask = torch.from_numpy(mask).to(src_B.device)

        with torch.no_grad():
            pred_tgt_B = self.G(src_B)[:, mask, :, :]
            tgt_B = tgt_B[:, mask, :, :]
            p = self.pearson_metric(pred_tgt_B.flatten(), tgt_B.flatten())
            self.pearson_tst.append(p)

