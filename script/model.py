from argparse import ArgumentParser
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

import torchmetrics
import pytorch_lightning as pl
import wandb

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



class LitTransferUnet(LitI2IGAN):
    def _init_models(self):
        old_dict = torch.load(self.hparams.pretrained_unet_path)
        state_dict = {}
        for key, value in old_dict.items():
            new_key = key.replace('module.', '')  # Remove "module." from the key
            state_dict[new_key] = value
        
        self.G_A = define_G(self.hparams.in_nc, self.hparams.out_nc, 
                            self.hparams.ngf, "unet_256", norm="batch", 
                            use_dropout=not self.hparams.no_dropout_G)
        self.G_A.load_state_dict(state_dict)
        for p in self.G_A.parameters():
            p.requires_grad = False

        self.G_transfer = define_G(self.hparams.in_nc, self.hparams.out_nc, 
                    self.hparams.ngf, "unet_256", norm="batch", 
                    use_dropout=not self.hparams.no_dropout_G)
        self.G_transfer.load_state_dict(state_dict)

        self.num_steps = 0
        self.num_val_steps = 0
        
        self.actuals = []
        self.sq_error = []

        self.actuals_val = []
        self.sq_error_val = []

        self.cossum = 0
        self.cossum_val = 0


    
    
    
    def configure_optimizers(self):
        opt_G = optim.AdamW(self.G_transfer.parameters(), lr=self.hparams.lr_G, 
                            betas=(self.hparams.b1_G, self.hparams.b2_G),
                            weight_decay=self.hparams.weight_decay_G)

        def lambda_rule(epoch):
            n_epochs_decay = float(100 - self.hparams.n_epochs_const + 1) #self.hparams.max_epochs
            lr_l = 1.0 - max(0, epoch - self.hparams.n_epochs_const) / n_epochs_decay
            return lr_l

        sch_G = lr_scheduler.LambdaLR(opt_G, lr_lambda=lambda_rule)

        return [[opt_G], [sch_G]]

    def training_step(self, batch, batch_idx):
        src_A, src_B = batch

        
        embed_A = self.G_A(src_A, layer_n=self.hparams.adaptation_layer)
        embed_B = self.G_transfer(src_B, layer_n=self.hparams.adaptation_layer)

        loss = nn.MSELoss()

        flat_A = torch.flatten(embed_A).detach().cpu().numpy()
        flat_B = torch.flatten(embed_B).detach().cpu().numpy()


        self.num_steps += 1

        if(self.num_steps % 111 < 60):
            self.cossum += np.dot(flat_A,flat_B)/(np.linalg.norm(flat_A)*np.linalg.norm(flat_B))
            self.sq_error.append((flat_A - flat_B)**2) 
            self.actuals.append(flat_A)
        
        else:
            self.cossum_val += np.dot(flat_A,flat_B)/(np.linalg.norm(flat_A)*np.linalg.norm(flat_B))
            self.sq_error_val.append((flat_A - flat_B)**2) 
            self.actuals_val.append(flat_A)

            
        if(self.num_steps % 111 == 59):
            vars = np.var(np.transpose(np.array(self.actuals)))
            errors = []
            for error in self.sq_error:
                errors.append(np.mean(error/vars))
            print("Training Set | epoch:", (self.num_steps+52)/111, "avg:", sum(errors)/59, "cos:", self.cossum/59)
            self.sq_error = []
            self.actuals = []
            self.cossum = 0
        
        if(self.num_steps % 111 == 0):
            vars = np.var(np.transpose(np.array(self.actuals_val)))
            errors = []
            for error in self.sq_error_val:
                errors.append(np.mean(error/vars))
            print("Test Set | epoch:", self.num_steps/111, "avg:", sum(errors)/52, "cos:", self.cossum_val/52)
            self.sq_error_val = []
            self.actuals_val = []
            self.cossum_val = 0

        if(self.num_steps % 111 < 60):
            return(loss(embed_A, embed_B))
        else:
            return(0*loss(embed_A, embed_B))
    
    def validation_step(self, batch, batch_idx):
        src_A, src_B = batch
        
    def test_step(self, batch, batch_idx):
        src_B, tgt_B = batch

        mask = get_constant_dim_mask(tgt_B[0].detach().cpu().numpy())
        mask = torch.from_numpy(mask).to(src_B.device)

        with torch.no_grad():
            pred_tgt_B = self.G(src_B)[:, mask, :, :]
            tgt_B = tgt_B[:, mask, :, :]
            p = self.pearson_metric(pred_tgt_B.flatten(), tgt_B.flatten())
            self.pearson_tst.append(p)






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
        self.num_steps = 0

        self.weights = [np.array([.01,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])]
        self.grads = [np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])]
        self.losses = [np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])]

        self.weights_list = [
                    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
                    [0,0.3,0 ,0 ,0,0 ,0,0,0 ,0,0 ,0 ,0.7 ,0,0,0,],]
        self.weight = self.weights_list[self.hparams.weight_id]
        self.weight = [w/sum(self.weight) for w in self.weight]

        config = {"weights":self.weight}
        wandb.init(
            project="weighted",
            config=config,
        )

        print(self.weight)
        self.grads_final = []


    def softmax(self, x):
        return(np.exp(x)/np.exp(x).sum())
        
    def cos_similarity(self, v1, v2):
        return(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))

    def configure_optimizers(self):
        gen_optimizer = optim.Adam(self.G.parameters(), lr=self.hparams.lr_G)
        disc_optimizers = [optim.Adam(D.parameters(), lr=self.hparams.lr_D) for D in self.D_list]

        def lambda_rule(epoch):
            n_epochs_decay = float(100 - self.hparams.n_epochs_const + 1) #self.hparams.max_epochs
            lr_l = 1.0 - max(0, epoch - self.hparams.n_epochs_const) / n_epochs_decay
            return lr_l

        opts = disc_optimizers + [gen_optimizer]
        schs = [lr_scheduler.LambdaLR(opt, lr_lambda=lambda_rule) for opt in opts]

        return opts, schs

    def training_step(self, batch, batch_idx, optimizer_idx):
        src_A, src_B = batch
        # D
        if optimizer_idx < len(self.D_list):
            with torch.no_grad():
                tgt_A = self.G_A(src_A, layer_n=(optimizer_idx))
                tgt_B = self.G(src_B, layer_n=(optimizer_idx))

            pred_y = self.D_list[(optimizer_idx)](tgt_A)
            y_A = torch.ones_like(pred_y)
            loss_A = self.bce_logits(pred_y, y_A)
    
            pred_y = self.D_list[(optimizer_idx)](tgt_B)
            y_B = torch.zeros_like(pred_y)
            loss_B = self.bce_logits(pred_y, y_B)
    
            loss_d = (loss_A + loss_B) / 2
            self.log(f"loss_d:{optimizer_idx}", loss_d, prog_bar=True, logger=True)
            wandb.log({f"loss D {optimizer_idx}": loss_d})
            self.D_losses[(optimizer_idx)].append(loss_d.item())
            return loss_d

        # G
        elif optimizer_idx == len(self.D_list):
            layers = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
        
            weight = self.weight
            
            loss_g = 0
            scale = np.ones(len(layers))
            for layer in layers:
                tgt_B = self.G(src_B, layer_n=layer)
                pred_y = self.D_list[layer](tgt_B)
                y_A = torch.ones_like(pred_y, requires_grad=False)
                loss_g_l = self.bce_logits(pred_y, y_A)
                wandb.log({f"loss G {layer}": loss_g_l})
                loss_g += loss_g_l*weight[layer]
            wandb.log({f"loss G": loss_g})
            self.log("loss_g", loss_g, prog_bar=True, logger=True)
            return loss_g


            """
            self.num_steps += 1
            layers = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
            parameters = (0,1,2,5,8,11,14,17,20,23,26,29,32,35,38,39,40)
                if (layer == 15):
                    gl = np.zeros(len(layers))
                    dg = []
                    loss_g_l.backward(retain_graph=True)
                    for param in self.G.parameters():
                        if param.grad is not None:
                            grad_flat = np.array(param.grad.cpu().detach().flatten(), dtype=np.float32)
                            dg.append(grad_flat)

                    for i in range(0,len(parameters)-1):                            
                        layer_mag = 0
                        if parameters[i+1] <= len(dg):
                            if self.num_steps == 0:
                                self.grads_final.append(np.concatenate(dg[parameters[i]:parameters[i+1]]))
                            else:
                                self.grads_final[i] = .9*self.grads_final[i] +.1*np.concatenate(dg[parameters[i]:parameters[i+1]])

                    for i, grad in enum(self.grads_final):
                        gl[i] = np.norm(grad)
                        
                    self.weights.append(.95*self.weights[-1]+.05*gl)
                                        
                    self.G.zero_grad()"""

                                                

        else:
            raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        src_A, src_B = batch

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

