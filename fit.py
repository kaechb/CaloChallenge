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
from losses import hinge, least_squares, wasserstein, gradient_penalty
from helpers import CosineWarmupScheduler

rng = np.random.default_rng()
from helpers import *

# from metrics import *
from models import *


class MDMA(pl.LightningModule):
    def __init__(self, path="/", **kwargs):
        """This initializes the model and its hyperparameters, also some loss functions are defined here"""
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.opt = kwargs["opt"]
        self.lr_g = kwargs["lr_d"]
        self.lr_d = kwargs["lr_d"]
        self.gan = kwargs["gan"]
        self.stop_mean = kwargs["stop_mean"]
        self.gen_net = Gen(**kwargs)
        self.dis_net = Disc(**kwargs)
        self.gp_weight = 10
        self.relu = torch.nn.ReLU()
        self.dis_net_dict = {"average": False, "model": self.dis_net, "step": 0}
        self.gen_net_dict = {"average": False, "model": self.gen_net, "step": 0}
        if "mean_field_loss" in kwargs.keys():
            self.mean_field_loss = kwargs["mean_field_loss"]
        else:
            self.mean_field_loss = False
        self.i = 0
        if kwargs["gan"] in ["hinge","ls"]:

            self.target_real = torch.ones(kwargs["batch_size"], 1)
            self.target_fake = torch.zeros(kwargs["batch_size"], 1) if self.gan!="hinge" else -torch.ones(kwargs["batch_size"], 1)
            self.loss= hinge if self.gan=="hinge" else least_squares

        else:
            self.loss=wasserstein
            self.gp = not kwargs["spectralnorm"]
            self.target_real=None
            self.target_fake=None
        self.mse = nn.MSELoss()
        self.names = ["E", "x", "y", "z"]
        self.E_loss_mean = 0
        self.E_loss=kwargs["E_loss"]
        self.lambda_=kwargs["lambda"]
        self.d_loss_mean=0
        self.g_loss_mean=0
        self.turn_on_zero_gp=False
        self.freq=kwargs["freq"]
        self.head_start=0
        self.noise=kwargs["noise"]
        self.N=kwargs["N"]
        self.sched=True
        self.scale_E=False
        self.max_score=0.1
        self.min_score=-0.1
        if kwargs["name"] == "middle":
            self.num_z = 45
            self.num_alpha = 16
            self.num_R = 9
        elif kwargs["name"] == "big":
            self.num_z = 45
            self.num_alpha = 50
            self.num_R = 18



    def on_validation_epoch_start(self, *args, **kwargs):

        self.hists_real = []
        self.hists_fake = []

        self.scores_real=hist.Hist(hist.axis.Regular(100, self.min_score, self.max_score))
        self.scores_fake=hist.Hist(hist.axis.Regular(100, self.min_score, self.max_score))
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
        self.max_score=0
        self.min_score=-0

    def load_datamodule(self, data_module):
        """needed for lightning training to work, it just sets the dataloader for training and validation"""

        self.data_module = data_module
        self.scaler = data_module.scaler
        if not hasattr(self, "scale"):
            self.power_lambda=self.scaler.transfs[0].lambdas_[0]
            self.mean=self.scaler.transfs[0]._scaler.mean_[0]
            self.scale=self.scaler.transfs[0]._scaler.scale_[0]

        print("scaler values:",self.scale,self.mean,self.power_lambda)


    def transform(self,x):
        x=(x**self.power_lambda-1)/self.power_lambda
        return (x-self.mean)/self.scale

    def inverse_transform(self,x):
        return ((x*self.scale+self.mean)*self.power_lambda+1)**(1/self.power_lambda)




    def sampleandscale(self, batch, mask, cond, scale=False):
        """Samples from the generator and optionally scales the output back to the original scale"""
        mask = mask.bool()
        if not self.noise:
            with torch.no_grad():
                    z = torch.normal(torch.zeros(batch.shape[0], batch.shape[1], batch.shape[2], device=batch.device), torch.ones(batch.shape[0], batch.shape[1], batch.shape[2], device=batch.device))
            z[mask] = 0  # Since mean field is initialized by sum, we need to set the masked values to zero
        else:
            z=[batch.shape[0],batch.shape[1]]
        fake = self.gen_net(z, mask=mask, cond=cond, weight=False)
        fake[:, :, 0] = torch.nn.functional.relu(fake[:, :, 0] - self.min_E) + self.min_E
        fake[:, :, 0] =fake[:, :, 0] -torch.nn.functional.relu(fake[:, :, 0] - self.max_E)
        # if self.noise and not scale:
        #     fake=fake*(torch.ones_like(fake)+torch.normal(torch.zeros_like(fake),torch.ones_like(fake))*1e-5)
        if scale:
            fake_scaled = self.scaler.inverse_transform(fake)
            fake_scaled[mask] = 0  # set the masked values to zero
            if self.data_module.scale_E:
                return fake_scaled,fake, self.scaler.inverse_transform(batch)
            else:
                return fake_scaled,fake
        else:
            fake[mask] = 0  # set the masked values to zero
            return fake

    def interpolates(self, real_data, generated_data, mask, cond):
        """Calculates the gradient penalty loss for WGAN GP, interpolated events are matched eventwise"""
        batch_size = real_data.size()[0]
        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, device=real_data.device)
        alpha = alpha.expand_as(real_data)
        interpolated = alpha * real_data + (1 - alpha) * generated_data
        interpolated = Variable(interpolated, requires_grad=True)
        # Calculate probability of interpolated examples
        prob_interpolated, _ = self.dis_net(interpolated, mask=mask, cond=cond, weight=False)
        return interpolated, prob_interpolated
        # # Calculate gradients of probabilities with respect to examples
        # gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated, grad_outputs=torch.ones_like(prob_interpolated), create_graph=True, retain_graph=True)[0]
        # # Gradients have shape (batch_size, num_particles, featzres),
        # # so flatten to easily take norm per example in batch
        # gradients = gradients.view(batch_size, -1)
        # self.log("gp grads",gradients.norm(dim=1).mean())
        # # Derivatives of the gradient close to 0 can cause problems because of
        # # the square root, so manually calculate norm and add epsilon
        # gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
        # # Return gradient penalty

        # gp=self.gp_weight * (self.relu(gradients_norm - 1) ** 2).mean()

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
        sched_d = CosineWarmupScheduler(opt_d, 2000, 500 * 1000)
        sched_g = CosineWarmupScheduler(opt_g, 2000, 500 * 1000)
        return sched_d, sched_g

    def train_disc(self, batch, mask, opt_d, cond):
        """Trains the discriminator"""
        with torch.no_grad():
            fake= self.sampleandscale(batch=batch, mask=mask, cond=cond)
        batch[mask] = 0

        if self.mean_field_loss:
            pred_real, mean_field = self.dis_net(batch, mask=mask, weight=False, cond=cond)
            pred_fake, _ = self.dis_net(fake.detach(), mask=mask, weight=False, cond=cond)
        else:
            mean_field = None
            pred_real, _ = self.dis_net(batch, mask=mask, weight=False, cond=cond)
            pred_fake, _ = self.dis_net(fake.detach(), mask=mask, weight=False, cond=cond)
        d_loss = self.loss(y_real=pred_real.reshape(-1),y_fake=pred_fake.reshape(-1),critic=True)
        self.d_loss_mean = d_loss.item() * 0.01 + 0.99 * self.d_loss_mean

        if self.batch_idx%10==0:
            if pred_fake.min()<self.min_score or pred_real.min()<self.min_score:
                self.min_score=min(pred_fake.min(),pred_real.min())
            if pred_fake.max()>self.max_score:
                self.max_score=pred_fake.max()
        if (pred_fake != pred_fake).any() or (pred_real != pred_real).any():
            print("mask: ",mask.all(), "cond:", cond, "fake: ",(fake==fake).all())
            print("pred nan")
            return None
        if self.gan=="wgan":
            interpolates,pred_interpolates=self.interpolates(batch, fake, mask=mask, cond=cond)
            if self.gp:
                gp = gradient_penalty( pred_interpolates, interpolates,)
                d_loss=d_loss+gp
                self._log_dict["Training/gp"] = gp
        if d_loss != d_loss:print("d_loss nan");return None
        self.manual_backward(d_loss)
        if (self.batch_idx + 1) % self.N == 0:
            opt_d.step()
            opt_d.zero_grad()
        self._log_dict["Training/d_loss"] = self.d_loss_mean
        return mean_field

    def train_gen(self, batch, mask, opt_g, cond, mean_field=None):
        """Trains the generator"""
        fake = self.sampleandscale(batch=batch, mask=mask, cond=cond)
        if mean_field is not None:
            pred, mean_field_gen = self.dis_net(fake, mask=mask, weight=False, cond=cond)
            assert mean_field.shape == mean_field_gen.shape
            mean_field = self.mse(mean_field_gen, mean_field.detach()).mean()
            self._log_dict["Training/mean_field"] = mean_field
        else:
            pred,_ = self.dis_net(fake, mask=mask, weight=False, cond=cond)

        if self.batch_idx%10==0:
            if pred.max()>self.max_score:
                self.max_score=pred.max()
        g_loss = self.loss(y_real=None,y_fake=pred.reshape(-1),critic=False)
        if self.g_loss_mean is None:
            self.g_loss_mean = g_loss
        if g_loss != g_loss:print("g_loss nan");return None
        self.g_loss_mean = g_loss.item() * 0.01 + 0.99 * self.g_loss_mean
        self._log_dict["Training/g_loss"] = self.g_loss_mean
        if self.mean_field_loss:
            g_loss += mean_field
        # if self.E_loss:
        #     if not self.scale_E:
        #         response_fake=self.inverse_transform(fake[:,:,0]).sum(1).reshape(-1)/(cond[:,0]+10).exp()
        #         response_real=self.inverse_transform(batch[:,:,0]).sum(1).reshape(-1)/(cond[:,0]+10).exp()
        #         E_loss =  self.mse(response_real,response_fake)
        #         self.E_loss_mean = self.E_loss_mean * 0.99 + 0.01*E_loss.detach().item()
        #         self._log_dict["Training/E_loss"] = self.E_loss_mean
        #         g_loss += self.lambda_ *E_loss
        #     else:
        #         response_fake=self.inverse_transform(fake[:,:,0]).sum(1).reshape(-1)/self.inverse_transform(cond[:,0])
        #         response_real=self.inverse_transform(batch[:,:,0]).sum(1).reshape(-1)/self.inverse_transform(cond[:,0])
        #         E_loss =  self.mse(response_real,response_fake)
        #         # E_loss =torch.clamp(F.relu(E_loss-response_real.std()),max=1)
        #         self.E_loss_mean = self.E_loss_mean * 0.99 + 0.01*E_loss.detach().item()
        #         self._log_dict["Training/E_loss"] = self.E_loss_mean
        #         g_loss += self.lambda_ *E_loss
        self.manual_backward(g_loss)
        if (self.batch_idx + 1) % self.N == 0:
            opt_g.step()
            opt_g.zero_grad()
        if g_loss != g_loss:
            print("others in g nan")
            return None


    def training_step(self, batch,batch_idx):
        """simplistic training step, train discriminator and generator"""

        self.batch_idx=batch_idx
        self._log_dict = {}
        if batch_idx==0:
            alphas={"Model/alpha{}".format(i):self.gen_net.blocks[i].alpha.mean().item() for i in range(len(self.gen_net.blocks))}
            self._log_dict.update(**alphas)
        batch, mask, cond = batch[0], batch[1].bool(), batch[2]
        cond = torch.cat((cond.reshape(-1, 1), (~mask).float().sum(1).reshape(-1, 1)/4000), dim=-1).float()

        if self.global_step > 500000 and self.stop_mean:
            self.mean_field_loss = False
        if len(batch) == 1:
            return None
        opt_d, opt_g = self.optimizers()
        sched_d, sched_g = self.lr_schedulers()
        mean_field = self.train_disc(batch=batch, mask=mask, opt_d=opt_d, cond=cond)
        self.head_start+=1
        if self.global_step % (self.freq) == 0 and self.head_start>1000:
            self.train_gen(batch=batch, mask=mask, opt_g=opt_g, cond=cond, mean_field=mean_field)
        self.logger.log_metrics(self._log_dict, step=self.global_step)
        if self.sched:
            sched_d.step()
            sched_g.step()

    def validation_step(self, batch, batch_idx):
        """This calculates some important metrics on the hold out set (checking for overtraining)"""
        if self.global_step==0:
            self.power_lambda=self.scaler.transfs[0].lambdas_[0]
            self.mean=self.scaler.transfs[0]._scaler.mean_[0]
            self.scale=self.scaler.transfs[0]._scaler.scale_[0]
        self._log_dict = {}
        batch, mask, cond = batch[0], batch[1].bool(), batch[2]
        cond = torch.cat((cond.reshape(-1, 1), (~mask).float().sum(1).reshape(-1, 1)/4000), dim=-1).float()
        # if not self.data_module.scaled:
        #     cond=self.transform(cond)
        self.w1ps = []
        scores_real=self.dis_net(batch, mask=mask, weight=False, cond=cond)[0].reshape(-1)
        with torch.no_grad():
            if self.data_module.scale_E:
                fake_scaled,fake,batch = self.sampleandscale(batch=batch, mask=mask, cond=cond, scale=True)
            else:
                fake_scaled,fake = self.sampleandscale(batch=batch, mask=mask, cond=cond, scale=True)

            scores_fake=self.dis_net(fake, mask=mask, weight=False, cond=cond)[0].reshape(-1)
            self.scores_real.fill(scores_real.cpu().numpy())
            self.scores_fake.fill(scores_fake.cpu().numpy())
            unpadded_fake = fake_scaled[~mask].cpu().numpy()
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
                #self.log(self.names[i] + "_weighted", weighted_w1p, on_step=False, on_epoch=True)

        self.log("w1p", np.mean(w1ps), on_step=False, on_epoch=True)
        if self.min_w1p<0.003 and self.centered_gp:
            self.turn_on_zero_gp=True
        self.log("weighted w1p", np.mean(weighted_w1ps), on_step=False, on_epoch=True)
        try:
            if np.mean(w1ps) < self.min_w1p:
                self.plot = plotting_point_cloud(
                    step=self.global_step,
                    logger=self.logger,
                )
                self.plot.plot_ratio(self.hists_fake, self.hists_real, weighted=False)
                self.plot.plot_ratio(self.weighted_hists_fake, self.weighted_hists_real,weighted=True)
                self.plot.plot_scores(self.scores_fake, self.scores_real )
                self.min_w1p = np.mean(w1ps)
                self.log("min_w1p", np.mean(w1ps), on_step=False, on_epoch=True)
                self.log("min_weighted_w1p", np.mean(w1ps), on_step=False, on_epoch=True)
        except:
            traceback.print_exc()
