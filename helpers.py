import numpy as np
import torch
from torch import optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math
import random
import warnings
import copy
import torch
import torch.nn as nn
from functools import partial
from math import sqrt
from torch.nn  import Parameter
import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm
from torch.nn import Parameter
import math
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm
from torch.nn import Parameter
import math
import matplotlib.pyplot as plt
import os
import mplhep as hep
import torch
import numpy as np
import hist
from hist import Hist
import traceback
import pandas as pd
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from helpers import mass
from scipy import stats
import datetime
import time
from torch import nn
from torch.nn import functional as FF
import traceback
import os
import pytorch_lightning as pl
import seaborn as sns
import hist
from hist import Hist
import matplotlib as mpl
import matplotlib.patches as mpatches
import torch.nn.functional as F
def mass(p, canonical=False):
    if not torch.is_tensor(p):
        p=torch.tensor(p)
    if len(p.shape)==2:
        n_dim = p.shape[1]
        p = p.reshape(-1, n_dim // 3, 3)

    px = torch.cos(p[:, :, 1]) * p[:, :, 2]
    py = torch.sin(p[:, :, 1]) * p[:, :, 2]
    pz = torch.sinh(p[:, :, 0]) * p[:, :, 2]
    #E = torch.sqrt(px**2 + py**2 + pz**2)
    E = p[:,:,3:].sum(axis=1)

    p = px.sum(axis=1) ** 2 + py.sum(axis=1) ** 2 + pz.sum(axis=1) ** 2
    m2 = E - p

    return torch.sqrt(torch.max(m2, torch.zeros(len(E)).to(E.device)))



class TPReLU(nn.Module):

    def __init__(self, num_parameters=1, init=0.25):
        self.num_parameters = num_parameters
        super(TPReLU, self).__init__()
        self.weight = Parameter(torch.Tensor(num_parameters).fill_(init))
        self.bias = Parameter(torch.zeros(num_parameters))

    def forward(self, input):
        bias_resize = self.bias.view(1, self.num_parameters, *((1,) * (input.dim() - 2))).expand_as(input)
        return F.prelu(input - bias_resize, self.weight.clamp(0, 1)) + bias_resize

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.num_parameters) + ')'
class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * min(self.max_num_iters*0.99,epoch) / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor



def to_canonical(data, rev=False):
    n_dim = data.shape[1]
    p = torch.zeros_like(data)
    if rev:
        p[:, :, 0] = torch.arctanh(data[:, :, 2]/ torch.sqrt(data[:, :, 0] ** 2 + data[:, :, 1] ** 2 + data[:, :, 2] ** 2))
        p[:, :, 1] = torch.atan2(data[:, :, 1], data[:, :, 0])
        p[:, :, 2] = torch.sqrt(data[:, :, 0] ** 2 + data[:, :, 1] ** 2)
        return p
    else:

        p[:, :, 0] = data[:, :, 2] * torch.cos(data[:, :, 1])
        p[:, :, 1] = data[:, :, 2] * torch.sin(data[:, :, 1])
        p[:, :, 2] = data[:, :, 2] * torch.sinh(data[:, :, 0])
        E=p[:,:,0]**2+p[:,:,1]**2+p[:,:,2]**2
        return torch.cat((p,E.reshape(data.shape[0],data.shape[1],1)),dim=-1)



from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class WeightNormalizedLinear(nn.Module):

    def __init__(self, in_features, out_features, scale=False, bias=False, init_factor=1, init_scale=1):
        super(WeightNormalizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        if scale:
            self.scale = Parameter(torch.Tensor(out_features).fill_(init_scale))
        else:
            self.register_parameter('scale', None)

        self.reset_parameters(init_factor)

    def reset_parameters(self, factor):
        stdv = 1. * factor / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def weight_norm(self):
        return self.weight.pow(2).sum(1).sqrt().add(1e-8)

    def norm_scale_bias(self, input):
        output = input.div(self.weight_norm().unsqueeze(0))
        if self.scale is not None:
            output = output.mul(self.scale.unsqueeze(0))
        if self.bias is not None:
            output = output.add(self.bias.unsqueeze(0))
        return output

    def forward(self, input):
        return self.norm_scale_bias(F.linear(input, self.weight))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

def masked_layer_norm(x, mask, eps = 1e-5):
    """
    x of shape: [batch_size (N), num_objects (L), features(C)]
    mask of shape: [batch_size (N), num_objects (L)]
    """
    mask = mask.float().unsqueeze(-1)  # (N,L,1)
    mean = (torch.sum(x * mask, 1) / torch.sum(mask, 1))   # (N,C)
    mean = mean.detach()
    var_term = ((x - mean.unsqueeze(1).expand_as(x)) * mask)**2  # (N,L,C)
    var = (torch.sum(var_term, 1) / torch.sum(mask, 1))  #(N,C)
    var = var.detach()
    mean_reshaped = mean.unsqueeze(1).expand_as(x)  # (N, L, C)
    var_reshaped = var.unsqueeze(1).expand_as(x)    # (N, L, C)
    ins_norm = (x - mean_reshaped) / torch.sqrt(var_reshaped + eps)   # (N, L, C)
    return ins_norm


def center_jets(data):   # assumse [batch, particles, features=[pt,y,phi])
    etas = jet_etas(data)  # pseudorapdityt
    phis = jet_phis(data)  # azimuthal angle
    etas = etas[:,np.newaxis].repeat(repeats=data.shape[1], axis=1)
    phis = phis[:,np.newaxis].repeat(repeats=data.shape[1], axis=1)
    mask = data[...,0] > 0   # mask all particles with nonzero pt
    data[mask,1] -= etas[mask]
    data[mask,2] -= phis[mask]
    return data


# fixed centering of the jets
def center_jets_tensor(data):   # assumse [batch, particles, features=[pt,y,phi])
    etas = jet_etas(data)  # pseudorapdityt
    phis = jet_phis(data)  # azimuthal angle
    etas = etas[:,np.newaxis].expand(-1,data.shape[1])
    phis = phis[:,np.newaxis].expand(-1,data.shape[1])
    mask = data[...,0] > 0   # mask all particles with nonzero pt
    data[...,1][mask] -= etas[mask]   # there is a bug here when calculating gradients
    data[...,2][mask] -= phis[mask]
    return data

def torch_p4s_from_ptyphi(ptyphi):
    # get pts, ys, phis
    #ptyphi = torch.Tensor(ptyphi).float()
    pts, ys, phis = (ptyphi[...,0,np.newaxis],
                     ptyphi[...,1,np.newaxis],
                     ptyphi[...,2,np.newaxis])

    Ets = torch.sqrt(pts**2) #  + ms**2) # everything assumed massless
    p4s = torch.cat((Ets*torch.cosh(ys), pts*torch.cos(phis),
                          pts*torch.sin(phis), Ets*torch.sinh(ys)), axis=-1)
    return p4s


def torch_p4s_from_ptyphi(ptyphi):
    # get pts, ys, phis
    #ptyphi = torch.Tensor(ptyphi).float()
    pts, ys, phis = (ptyphi[...,0,np.newaxis],
                     ptyphi[...,1,np.newaxis],
                     ptyphi[...,2,np.newaxis])

    Ets = torch.sqrt(pts**2) #  + ms**2) # everything assumed massless
    p4s = torch.cat((Ets*torch.cosh(ys), pts*torch.cos(phis),
                          pts*torch.sin(phis), Ets*torch.sinh(ys)), axis=-1)
    return p4s


def jet_etas(jets_tensor):
    jets_p4s = torch_p4s_from_ptyphi(jets_tensor)
    etas = torch_etas_from_p4s(jets_p4s.sum(axis=1))
    return etas

def jet_phis(jets_tensor):
    jets_p4s = torch_p4s_from_ptyphi(jets_tensor)
    phis = torch_phis_from_p4s(jets_p4s.sum(axis=1), phi_ref=0)
    return phis

def torch_etas_from_p4s(p4s):
    ## PSEUDO-RAPIDITY
    out = torch.zeros(p4s.shape[:-1],device=p4s.device).float()
    nz_mask = torch.any(p4s != 0., axis=-1)
    nz_p4s = p4s[nz_mask]
    out[nz_mask] = torch.atanh(nz_p4s[...,3]/torch.sqrt(nz_p4s[...,1]**2 + nz_p4s[...,2]**2 + nz_p4s[...,3]**2))
    return out


def torch_phi_fix(phis, phi_ref, copy=False):
    TWOPI = 2*np.pi
    diff = (phis - phi_ref)
    new_phis = torch.copy(phis) if copy else phis
    new_phis[diff > np.pi] -= TWOPI
    new_phis[diff < -np.pi] += TWOPI
    return new_phis


def torch_phis_from_p4s(p4s, phi_ref=None, _pts=None, phi_add2pi=True):
    # get phis
    phis = torch.atan2(p4s[...,2], p4s[...,1])
    if phi_add2pi:
        phis[phis<0] += 2*np.pi
    # ensure close to reference value
    if phi_ref is not None:
        phis = torch_phi_fix(phis, phi_ref, copy=False)

    return phis

class plotting_point_cloud():
    '''This is a class that takes care of  plotting steps in the script,
        It is initialized with the following arguments:
        true=the simulated data, note that it needs to be scaled
        gen= Generated data , needs to be scaled
        step=The current step of the training, this is need for tensorboard
        model=the model that is trained, a bit of an overkill as it is only used to access the losses
        config=the config used for training
        logger=The logger used for tensorboard logging'''
    def __init__(self,true,fake,mask,step=None,logger=None,weight=1):


        self.test_set=true.numpy()
        self.step=step
        self.fake=fake.numpy()
        self.mask=mask.numpy()
        self.weight=weight
        self.fig_size1=[6.4, 6.4]
        self.fig_size2=[2*6.4, 6.4]
        self.fig_size3=[3*6.4, 6.4]
        self.fig_size4=[4*6.4, 6.4]
        self.alpha=0.3
        mpl.rcParams['lines.linewidth'] = 2
        font = {"family": "normal", "size": 18}
        mpl.rc("font", **font)
        mpl.rc('lines', linewidth=2)
        sns.set_palette("Pastel1")
        if logger is not None:
            self.summary=logger
        else:
            self.summary = None

    def plot_mass(self,save=None,quantile=False,bins=15,plot_vline=False,title="",leg=-1):
        #This creates a histogram of the inclusive distributions and calculates the mass of each jet
        #and creates a histogram of that
        #if save, the histograms are logged to tensorboard otherwise they are shown
        #if quantile, this also creates a histogram of a subsample of the generated data,
        # where the mass used to condition the flow is in the first 10% percentile of the simulated mass dist
        i=0
        k=0
        fig,ax=plt.subplots(2,4,gridspec_kw={'height_ratios': [4, 1]},figsize=self.fig_size4)
        plt.suptitle("All Hits",fontsize=18)

        for v,name in zip(["eta","phi","pt","E"],[r"$\eta$",r"$\phi$",r"$p_T$",r"$E$"]):

            a=min(np.quantile(self.fake.reshape(-1,4)[:,i],0.001),np.quantile(self.test_set.reshape(-1,4)[:,i],0.001))
            b=max(np.quantile(self.fake.reshape(-1,4)[:,i],0.999),np.quantile(self.test_set.reshape(-1,4)[:,i],0.999))
            temp=self.test_set[:i]
            h=hist.Hist(hist.axis.Regular(bins,a,b))
            h2=hist.Hist(hist.axis.Regular(bins,a,b))
            h.fill(self.fake.reshape(-1,4)[self.fake.reshape(-1,4)[:,i]!=0,i])
            h2.fill(self.test_set.reshape(-1,4)[self.test_set.reshape(-1,4)[:,i]!=0,i])
            i+=1



            #hep.cms.label(data=False,lumi=None ,year=None,rlabel="",llabel="Private Work",ax=ax[0] )

            main_ax_artists, sublot_ax_arists = h.plot_ratio(
                h2,
                ax_dict={"main_ax":ax[0,k],"ratio_ax":ax[1,k]},
                rp_ylabel=r"Ratio",
                bar_="blue",
                rp_num_label="Generated",
                rp_denom_label="Ground Truth",
                rp_uncert_draw_type="line",  # line or bar
            )
            ax[0,k].set_xlabel("")


            # ax[0,k].patches[1].set_fc("orange")
            # ax[0,k].patches[1].set_alpha(0.5)
#                 if quantile and v=="m" and plot_vline:
#                     ax[0,k].hist(m[m_t<np.quantile(m_t,0.1)],histtype='step',bins=bins,alpha=1,color="red",label="10% quantile gen",hatch="/")
#                     ax[0,k].vlines(np.quantile(m_t,0.1),0,np.max(h[:]),color="red",label='10% quantile train')

            #ax[0,k].hist(temp,bins=bins,color="orange",alpha=0.5)
            ax[0,k].patches[1].set_fill(True)
            ax[0,k].ticklabel_format(axis="y",style="scientific",scilimits=(-3,3),useMathText=True)

            ax[0,k].patches[1].set_fc(sns.color_palette()[1])
            ax[0,k].patches[1].set_edgecolor("black")

            ax[0,k].patches[1].set_alpha(self.alpha)

            ax[1,k].set_ylim(0.25,2)
            ax[0,k].set_xlim(a,b)
            ax[1,k].set_xlabel(name)
            ax[1,k].set_xlim(a,b)
            ax[0,k].set_ylabel("Counts" )
            ax[1,k].set_ylabel("Ratio")
            ax[0,k].patches[0].set_lw(2)
            ax[0,k].get_legend().remove()
            k+=1
        ax[0,leg].legend(loc="best",fontsize=18)
        handles, labels = ax[0,leg].get_legend_handles_labels()
        ax[0,-1].locator_params(nbins=4,axis="x")
        ax[1,-1].locator_params(nbins=4,axis="x")
        handles[1]=mpatches.Patch(color=sns.color_palette()[1], label='The red data')
        ax[0,leg].legend(handles, labels)
        plt.tight_layout(pad=1)
        # if not save==None:
        #     plt.savefig(save+".pdf",format="pdf")
        if self.summary:

            plt.tight_layout()
            self.summary.log_image("inclusive", [fig],self.step)
            plt.close()
        else:
            plt.savefig("plots/inclusive_"+self.p+".pdf",format="pdf")
            plt.show()
    def plot_scores(self,pred_real,pred_fake,train,step):
        fig, ax = plt.subplots()

        bins=30#np.linspace(0,1,10 if train else 100)
        ax.hist(pred_fake, label="Generated", bins=bins, histtype="step")
        if pred_real.any():
            ax.hist(pred_real, label="Ground Truth", bins=bins, histtype="stepfilled",alpha=self.alpha)
        ax.legend()
        ax.patches[0].set_lw(2)
        plt.ylabel("Counts")
        plt.xlabel("Critic Score")
        if self.summary:
            plt.tight_layout()
            if pred_real.any():
                self.summary.log_image("class_train" if train else "class_val", [fig],self.step)
            else:
                self.summary.log_image("class_gen", [fig],self.step)
            plt.close()
        else:
            plt.savefig("plots/scores_"+str(train)+".pdf",format="pdf")
            plt.show()
# %%
import os
import numpy as np
import torch
import h5py
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler
from tqdm import tqdm

os.chdir("/home/mscham/calochallange/")
data_dir = "/home/mscham/fgsim/data/calochallange2/"


# %%
# creating instance of HighLevelFeatures class to handle geometry based on binning file
class ScalerBase:
    def __init__(self, transfs, featurenames: list[str]) -> None:
        self.transfs = transfs
        self.featurenames = featurenames
        self.n_features = len(self.transfs)

        self.scalerpath = Path(data_dir) / "scaler.gz"
        if self.scalerpath.is_file():
            self.transfs = joblib.load(self.scalerpath)

    def save_scalar(self, pcs: torch.Tensor):
        # The features need to be converted to numpy immediatly
        # otherwise the queuflow afterwards doesnt work
        assert pcs.dim() == 2
        assert self.n_features == pcs.shape[1]
        pcs = pcs.detach().cpu().numpy()
        self.plot_scaling(pcs)

        assert pcs.shape[1] == self.n_features
        pcs = np.hstack(
            [
                transf.fit_transform(arr.reshape(-1, 1))
                for arr, transf in zip(pcs.T, self.transfs)
            ]
        )
        self.plot_scaling(pcs, True)

        joblib.dump(self.transfs, self.scalerpath)

    def transform(self, pcs: np.ndarray):
        assert len(pcs.shape) == 2
        assert pcs.shape[1] == self.n_features
        return np.hstack(
            [
                transf.transform(arr.reshape(-1, 1))
                for arr, transf in zip(pcs.T, self.transfs)
            ]
        )

    def inverse_transform(self, pcs: torch.Tensor):
        assert pcs.shape[-1] == self.n_features
        orgshape = pcs.shape
        dev = pcs.device
        pcs = pcs.to("cpu").detach().reshape(-1, self.n_features).numpy()

        t_stacked = np.hstack(
            [
                transf.inverse_transform(arr.reshape(-1, 1))
                for arr, transf in zip(pcs.T, self.transfs)
            ]
        )
        return torch.from_numpy(t_stacked.reshape(*orgshape)).float().to(dev)

    def plot_scaling(self, pcs, post=False):
        for k, v in zip(self.featurenames, pcs.T):
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.hist(v, bins=500)
            fig.savefig(
                Path(data_dir) / f"{k}_post.png"
                if post
                else Path(data_dir) / f"{k}_pre.png"
            )
            plt.close(fig)


# %%

