import sys
import traceback

import numpy as np
import pytorch_lightning as pl
import torch
from scipy.stats import wasserstein_distance
from torch import nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torch.nn.functional import leaky_relu, sigmoid
from torch.nn.utils.rnn import pad_sequence
from torch.optim.swa_utils import AveragedModel

from helpers import CosineWarmupScheduler

rng = np.random.default_rng()
from helpers import *

# from metrics import *
from models import *


class MF(pl.LightningModule):
    def __init__(self, config, path="/", **kwargs):
        """This initializes the model and its hyperparameters, also some loss functions are defined here"""
        super().__init__()
        self.automatic_optimization = False
        self.opt = config["opt"]
        self.lr_g = config["lr_g"]
        self.lr_d = config["lr_d"]
        self.gan = kwargs["gan"]
        self.stop_mean = config["stop_mean"]
        self.gen_net = Gen(**config)
        self.dis_net = Disc(**config)
        self.true_fpd = None
        self.w1m_best = 0.2
        self.gp_weight = 10
        self.save_hyperparameters()
        self.relu = torch.nn.ReLU()
        self.dis_net_dict = {"average": False, "model": self.dis_net, "step": 0}
        self.gen_net_dict = {"average": False, "model": self.gen_net, "step": 0}
        if "mean_field_loss" in config.keys():
            self.mean_field_loss = config["mean_field_loss"]
        else:
            self.mean_field_loss = False
        self.i = 0
        self.g_loss_mean = 0.5
        self.d_loss_mean = 0.5
        self.target_real = torch.ones(config["batch_size"], 1)
        self.target_fake = torch.zeros(config["batch_size"], 1)
        self.mse = nn.MSELoss()
        self.names = ["E", "z", "alpha", "R"]
        if config["name"] == "middle":
            self.num_z = 45
            self.num_alpha = 16
            self.num_R = 9
        elif config["name"] == "big":
            self.num_z = 45
            self.num_alpha = 50
            self.num_R = 18
        if not hasattr(self, "min_E"):
            self.min_E = 10000
        self.E_loss_mean = 0
        self.E_loss_fake_mean = 0

    def on_validation_epoch_start(self, *args, **kwargs):
        # self.dis_net = self.dis_net.cpu()
        # self.gen_net = self.gen_net.cpu()
        self.hists_real = []
        self.hists_fake = []
        self.weighted_hists_real = []
        self.weighted_hists_fake = []
        self.hists_real.append(hist.Hist(hist.axis.Regular(100, 0, 6000)))
        self.hists_fake.append(hist.Hist(hist.axis.Regular(100, 0, 6000)))
        for n in [self.num_z, self.num_alpha, self.num_R]:
            self.hists_real.append(hist.Hist(hist.axis.Integer(0, n)))
            self.hists_fake.append(hist.Hist(hist.axis.Integer(0, n)))

            self.weighted_hists_real.append(hist.Hist(hist.axis.Integer(0, n)))
            self.weighted_hists_fake.append(hist.Hist(hist.axis.Integer(0, n)))
        self.gen_net.eval()
        self.dis_net.train()

    def on_validation_epoch_end(self, *args, **kwargs):
        self.gen_net.train()
        self.dis_net.train()

    def load_datamodule(self, data_module):
        """needed for lightning training to work, it just sets the dataloader for training and validation"""
        self.data_module = data_module
        self.scaler = data_module.scaler

    def sampleandscale(self, batch, mask, cond, scale=False):
        """Samples from the generator and optionally scales the output back to the original scale"""
        mask = mask.bool()
        with torch.no_grad():
            z = torch.normal(torch.zeros(batch.shape[0], batch.shape[1], batch.shape[2], device=batch.device), torch.ones(batch.shape[0], batch.shape[1], batch.shape[2], device=batch.device))
        z[mask] = 0  # Since mean field is initialized by sum, we need to set the masked values to zero
        fake, E = self.gen_net(z, mask=mask, cond=cond, weight=False)
        fake[:, :, 0] = torch.nn.functional.relu(fake[:, :, 0] - self.min_E) + self.min_E
        if scale:
            fake_scaled = self.scaler.inverse_transform(fake)

            fake_scaled[mask] = 0  # set the masked values to zero
            return fake_scaled
        else:
            fake[mask] = 0  # set the masked values to zero
            return fake, E

    def _gradient_penalty(self, real_data, generated_data, mask, cond):
        """Calculates the gradient penalty loss for WGAN GP, interpolated events are matched eventwise"""
        batch_size = real_data.size()[0]
        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, device=real_data.device)
        alpha = alpha.expand_as(real_data)
        interpolated = alpha * real_data + (1 - alpha) * generated_data
        interpolated = Variable(interpolated, requires_grad=True)
        # Calculate probability of interpolated examples
        prob_interpolated, _, _ = self.dis_net(interpolated, mask=mask, cond=cond, weight=False)
        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated, grad_outputs=torch.ones_like(prob_interpolated), create_graph=True, retain_graph=True)[0]
        # Gradients have shape (batch_size, num_particles, featzres),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def configure_optimizers(self):
        """Sets the optimizer and the learning rate scheduler"""
        if self.opt == "Adam":
            opt_g = torch.optim.Adam(self.gen_net.parameters(), lr=self.lr_g, betas=(0.0, 0.999), eps=1e-14)
            opt_d = torch.optim.Adam(self.dis_net.parameters(), lr=self.lr_d, betas=(0.0, 0.999), eps=1e-14)  #
        elif self.opt == "AdamW":
            opt_g = torch.optim.Adam(self.gen_net.parameters(), lr=self.lr_g, betas=(0.0, 0.999), eps=1e-14)
            opt_d = torch.optim.AdamW(self.dis_net.parameters(), lr=self.lr_d, betas=(0.0, 0.999), eps=1e-14, weight_decay=0.01)  #
        else:
            raise
        sched_d, sched_g = self.schedulers(opt_d, opt_g)
        return [opt_d, opt_g], [sched_d, sched_g]

    def schedulers(self, opt_d, opt_g):
        sched_d = CosineWarmupScheduler(opt_d, 200, 1000 * 1000)
        sched_g = CosineWarmupScheduler(opt_g, 200, 1000 * 1000)
        return sched_d, sched_g

    def train_disc(self, batch, mask, opt_d, cond):
        """Trains the discriminator"""
        with torch.no_grad():
            fake, _ = self.sampleandscale(batch=batch, mask=mask, cond=cond)
        opt_d.zero_grad()
        self.dis_net.zero_grad()
        batch[mask] = 0
        if self.mean_field_loss:
            pred_real, mean_field, E = self.dis_net(batch, mask=mask, weight=False, cond=cond)  # mean_field is used for feature matching
            pred_fake, _, Efake = self.dis_net(fake.detach(), mask=mask, weight=False, cond=cond)
        else:
            mean_field = None
            pred_real, _, E = self.dis_net(batch, mask=mask, weight=False, cond=cond)
            pred_fake, _, Efake = self.dis_net(fake.detach(), mask=mask, weight=False, cond=cond)
        pred_fake = pred_fake.reshape(-1)
        pred_real = pred_real.reshape(-1)
        if (pred_fake != pred_fake).any() or (pred_real != pred_real).any():
            return None
        if self.gan == "ls":
            target_fake = torch.zeros_like(pred_fake)
            target_real = torch.ones_like(pred_real)
            d_loss = self.mse(pred_fake, target_fake).mean() + self.mse(pred_real, target_real).mean()

            self.d_loss_mean = d_loss.detach() * 0.01 + 0.99 * self.d_loss_mean
        elif self.gan == "hinge":
            d_loss = F.relu(1 - pred_real).mean() + F.relu(1 + pred_fake).mean()

            self.d_loss_mean = d_loss.detach() * 0.01 + 0.99 * self.d_loss_mean
        else:
            gp = self._gradient_penalty(batch, fake, mask=mask, cond=cond)
            d_loss = -pred_real.mean() + pred_fake.mean()

            self.d_loss_mean = d_loss.detach() * 0.01 + 0.99 * self.d_loss_mean
            d_loss += gp
            self._log_dict["Training/gp"] = gp
        E_loss = 1e-5 * self.mse(E.reshape(-1), cond[:, 0].reshape(-1)) + self.mse(Efake.reshape(-1), cond[:, 0].reshape(-1))
        self.E_loss_mean = self.E_loss_mean * 0.99 + E_loss.detach().item()
        self._log_dict["Training/E_loss"] = self.E_loss_mean
        d_loss += E_loss
        self.manual_backward(d_loss)
        opt_d.step()
        self._log_dict["Training/lr_d"] = opt_d.param_groups[0]["lr"]
        self._log_dict["Training/d_loss"] = self.d_loss_mean
        return mean_field

    def train_gen(self, batch, mask, opt_g, cond, mean_field=None):
        """Trains the generator"""
        opt_g.zero_grad()
        self.gen_net.zero_grad()
        fake, E = self.sampleandscale(batch=batch, mask=mask, cond=cond)
        if mask is not None:
            fake = fake * (~mask).unsqueeze(-1)
        if mean_field is not None:
            pred, mean_field_gen, Efake = self.dis_net(fake, mask=mask, weight=False, cond=cond)
            assert mean_field.shape == mean_field_gen.shape
            mean_field = self.mse(mean_field_gen, mean_field.detach()).mean()
            self._log_dict["Training/mean_field"] = mean_field
        else:
            pred, _, Efake = self.dis_net(fake, mask=mask, weight=False, cond=cond)
        pred = pred.reshape(-1)
        if self.gan == "ls":
            target = torch.ones_like(pred)
            g_loss = 0.5 * self.mse(pred, target).mean()
        else:
            g_loss = -pred.mean()
        if g_loss != g_loss or mean_field != mean_field:
            return None
        if self.g_loss_mean is None:
            self.g_loss_mean = g_loss
        self.g_loss_mean = g_loss.detach() * 0.01 + 0.99 * self.g_loss_mean
        if self.mean_field_loss:
            g_loss += mean_field
        E_loss = 1e-5 * self.mse(E.reshape(-1), cond[:, 0].reshape(-1))
        self.E_loss_fake_mean = self.E_loss_fake_mean * 0.99 + E_loss.detach().item()
        self._log_dict["Training/E_loss_fake"] = self.E_loss_fake_mean
        self.manual_backward(g_loss)
        opt_g.step()

        self._log_dict["Training/lr_g"] = opt_g.param_groups[0]["lr"]
        self._log_dict["Training/g_loss"] = self.g_loss_mean

    def training_step(self, batch):
        """simplistic training step, train discriminator and generator"""
        batch, mask, cond = batch[0], batch[1].bool(), batch[2]
        cond = torch.cat((cond.reshape(-1, 1), (~mask).float().sum(1).reshape(-1, 1)), dim=-1).float()
        if self.current_epoch < 1:
            if batch[:, :, 0].min() < self.min_E:
                self.min_E = batch[:, :, 0].min()
        if not hasattr(self, "freq"):
            self.freq = 1
        self._log_dict = {}
        if self.global_step > 500000 and self.stop_mean:
            self.mean_field_loss = False
        if len(batch) == 1:
            return None

        opt_d, opt_g = self.optimizers()
        sched_d, sched_g = self.lr_schedulers()
        ### GAN PART
        mean_field = self.train_disc(batch=batch, mask=mask, opt_d=opt_d, cond=cond)
        if self.global_step % (self.freq) == 0:
            self.train_gen(batch=batch, mask=mask, opt_g=opt_g, cond=cond, mean_field=mean_field)
            self.i += 1
            if self.i % (100 // self.freq) == 0:
                self.logger.log_metrics(self._log_dict, step=self.global_step)
        sched_d.step()
        sched_g.step()

    def validation_step(self, batch, batch_idx):
        """This calculates some important metrics on the hold out set (checking for overtraining)"""
        self._log_dict = {}

        batch, mask, cond = batch[0], batch[1].bool(), batch[2]
        cond = torch.cat((cond.reshape(-1, 1), (~mask).float().sum(1).reshape(-1, 1)), dim=-1).float()
        self.w1ps = []
        if batch[:, :, 0].min() < self.min_E:
            self.min_E = batch[:, :, 0].min()
        with torch.no_grad():
            fake = self.sampleandscale(batch=batch, mask=mask, cond=cond, scale=True)
            # fake=fake.clamp_(min=self.min_E,out=fake)

            # scores_real = self.dis_net(batch, mask=mask[:len(mask)], weight=False)[0]
            # scores_fake = self.dis_net(f, mask=mask, weight=False )[0]
            unpadded_fake = fake[~mask].cpu().numpy()
            unpadded_real = batch[~mask].cpu().numpy()
            for i in range(4):
                self.hists_fake[i].fill(unpadded_fake[:, i].reshape(-1))
                self.hists_real[i].fill(unpadded_real[:, i].reshape(-1))
            for i in range(3):
                self.weighted_hists_fake[i].fill(unpadded_fake[:, i + 1].reshape(-1), weight=unpadded_fake[:, 0].reshape(-1))
                self.weighted_hists_real[i].fill(unpadded_real[:, i + 1].reshape(-1), weight=unpadded_real[:, 0].reshape(-1))

    def on_validation_epoch_end(self):
        w1ps = []
        weighted_w1ps = []
        if not hasattr(self, "min_w1p"):
            self.min_w1p = 10
        for i in range(4):
            cdf_fake = self.hists_fake[i].values().cumsum()
            cdf_real = self.hists_real[i].values().cumsum()
            cdf_fake /= cdf_fake[-1]
            cdf_real /= cdf_real[-1]
            w1p = np.mean(np.abs(cdf_fake - cdf_real))
            w1ps.append(w1p)
            self.log(self.names[i], w1p, on_step=False, on_epoch=True)
            if i > 0:
                weighted_cdf_fake = self.hists_fake[i].values().cumsum()
                weighted_cdf_real = self.hists_real[i].values().cumsum()
                weighted_cdf_fake /= weighted_cdf_fake[-1]
                weighted_cdf_real /= weighted_cdf_real[-1]
                weighted_w1p = np.mean(np.abs(weighted_cdf_fake - weighted_cdf_real))
                weighted_w1ps.append(weighted_w1p)
                self.log(self.names[i] + "_weighted", weighted_w1p, on_step=False, on_epoch=True)
        self.log("w1p", np.mean(w1ps), on_step=False, on_epoch=True)
        self.log("weighted w1p", np.mean(weighted_w1ps), on_step=False, on_epoch=True)

        try:
            if np.mean(w1ps) < self.min_w1p:
                self.plot = plotting_point_cloud(
                    step=self.global_step,
                    logger=self.logger,
                )
                self.plot.plot_ratio(self.hists_fake, self.hists_real, weighted=False)
                self.plot.plot_ratio(self.weighted_hists_fake, self.weighted_hists_real, weighted=True)
                self.min_w1p = np.mean(w1ps)
                self.log("min_w1p", np.mean(w1ps), on_step=False, on_epoch=True)
                self.log("min_weighted_w1p", np.mean(w1ps), on_step=False, on_epoch=True)
        except:
            traceback.print_exc()
