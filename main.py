import datetime
import os
import sys
import time
import traceback

import wandb



import numpy as np
import pandas as pd
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.tuner.tuning import Tuner
from scipy import stats
from torch.nn import functional as FF
import torch
# from pytorch_lightning.plugins.environments import SLURMEnvironment
from helpers import EqualLR
from dataloader import PointCloudDataloader
from fit import MDMA
from tqdm import tqdm

from preprocess import DQ, DQKDE, Cart, LogTransformer, ScalerBase
# from plotting import plotting


def train(config,data_module, ckpt=False,logger=None):
    torch.set_float32_matmul_precision('medium' )
    # This function is a wrapper for the hyperparameter optimization module called ray
    # Its parameters hyperopt and ckpt are there for convenience
    # Config is the only relevant parameter as it sets the trainings hyperparameters
    # hyperopt:whether to optimizer hyper parameters - ckpt: path to checkpoint if used
    # Callbacks to use during the training, we  checkpoint our models

    if not ckpt:
        model = MDMA( **config)
        model.continue_training=False
        model.ckpt=None
    else:
        print("model loaded")
        print("ckpt: ",ckpt)
        model=MDMA.load_from_checkpoint(ckpt)
        model.ckpt=ckpt
        model.d_loss_mean=0.5
        model.g_loss_mean=0.5
        model.mean_field_loss=config["mean_field_loss"]
        if config["name"]=="big":
            model.num_z = 45
            model.num_alpha = 50
            model.num_R = 18
        model.E_loss=True
    # Set up the dataloader
    minE=10
    maxE=-10
    for i in data_module.train_dataloader():
        if i[0][i[0][:,:,0]!=0][:,0].min()<minE:
            model.min_E=i[0][:,:,0].min()
        if i[0][i[0][:,:,0]!=0][:,0].max()>maxE:
            model.max_E=i[0][:,:,0].max()
    model.load_datamodule(data_module)
    model.head_start=0
    callbacks = [ModelCheckpoint(monitor="weighted w1p", save_top_k=3, mode="min",filename="{epoch}-{w1p:.5f}-{E:.7f}",every_n_epochs=1,),pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ModelCheckpoint(monitor="E", save_top_k=3, mode="min",filename="{epoch}-{w1p:.5f}-{E:.7f}",every_n_epochs=1,)]
    # the sets up the model, with some options that can be set
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        logger=logger,
        log_every_n_steps=100,
        max_epochs=20000,
        callbacks=callbacks,
        val_check_interval=1000,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=1,
        enable_progress_bar=False,
        default_root_dir="/beegfs/desy/user/{}/calochallenge".format(os.environ["USER"]),

    )
    torch.autograd.set_detect_anomaly(True)
    print(trainer.default_root_dir)
    # This calls the fit function which trains the model
    print("This is run: ", logger.experiment.name)
    if ckpt:
        trainer.fit(model,datamodule=data_module,ckpt_path=ckpt)
    else:
        trainer.fit(model,datamodule=data_module,)


if __name__ == "__main__":


    config = {
        "batch_size": 128,
        "dropout": 0.,
        "gan": "wgan",
        "heads": 2,
        "heads_gen": 16,
        "hidden_gen": 64,
        "hidden": 32,
        "l_dim_gen": 16,
        "l_dim": 16,
        "lr_d": 0.0001,
        "lr_g": 0.0001,
        "max_epochs": 2000,
        "n_dim": 4,
        "num_layers_gen": 8,
        "num_layers": 2,
        "opt": "Adam",
        "mean_field_loss":True,
        "stop_mean":True,
        "ckpt":False,
        "freq":1,
        "name":"middle",
        "max":True,
        "lambda":0.1,
        "E_loss":True,
        "cartesian":True,
        "N":1,
        "sched":True,
        "cond_dim":2,
        "scale_E":True,
        "equallr":False,
        "spectralnorm":False,
        "noise":False,

    }
    #set up WandB logger
    logger = WandbLogger(
        save_dir="/beegfs/desy/user/{}/calochallenge".format(os.environ["USER"]),
        sync_tensorboard=False,
        project="CaloChallenge2")
    #best function that exists in ml
    logger.experiment.log_code(".")
    # update config with hyperparameters from sweep
    if len(logger.experiment.config.keys()) > 0:
        config.update(**logger.experiment.config)
    config["l_dim_gen"]=config["l_dim"]
    print(logger.experiment.dir)
    print("config:", config)
    # if not config["ckpt"]:
    #     ckpt=False
    # if config["name"]=="middle":
    #     ckpt="/beegfs/desy/user/kaechben/calochallenge/CaloChallenge/xj3x18wx/checkpoints/epoch=462-w1p=0.00077-E=0.0002544.ckpt"
    # if config["name"]=="big":
    #     ckpt="/beegfs/desy/user/kaechben/calochallenge/CaloChallenge/jx98l14b/checkpoints/epoch=300-w1p=0.00097-E=0.0000770.ckpt"
    #ckpt="/beegfs/desy/user/{}/pf_t/linear/mao7f3bq/checkpoints/epoch=5513-w1m=0.00020.ckpt"
    data_module = PointCloudDataloader(name=config["name"],batch_size=config["batch_size"],max=config["max"],cartesian=config["cartesian"],scale_E=config["scale_E"])
    data_module.setup("fit")
    minE=10
    maxE=-10

    train(config,data_module,logger=logger,ckpt=False)  # load_ckpt=ckptroot=root,