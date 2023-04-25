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
from helpers import *
from dataloader import PointCloudDataloader
from fit import MF
from tqdm import tqdm
# from plotting import plotting


# from comet_ml import Experiment
def lcm(a, b):
    return (a * b) // math.gcd(a, b)


def train(config, ckpt=False,logger=None):
    torch.set_float32_matmul_precision('medium' )
    # This function is a wrapper for the hyperparameter optimization module called ray
    # Its parameters hyperopt and ckpt are there for convenience
    # Config is the only relevant parameter as it sets the trainings hyperparameters
    # hyperopt:whether to optimizer hyper parameters - ckpt: path to checkpoint if used
    # Callbacks to use during the training, we  checkpoint our models

    if not ckpt:
        model = MF( config=config, **config)
        model.continue_training=False
        model.ckpt=None
    else:
        print("model loaded")
        print("ckpt: ",ckpt)
        model=MF.load_from_checkpoint(ckpt)
        model.ckpt=ckpt
        model.d_loss_mean=0.5
        model.g_loss_mean=0.5
        model.mean_field_loss=config["mean_field_loss"]
        # torch.nn.init.kaiming_normal_(model.dis_net.out.weight)
        # torch.nn.init.kaiming_normal_(model.gen_net.out.weight)
        logger.log_hyperparams({"mean_field_loss":model.mean_field_loss,"ckpt":model.ckpt,"lr_g":config["lr_g"],"lr_d":config["lr_d"]})


    model.lr_g=config["lr_g"]
    model.lr_d=config["lr_g"]
    # Set up the dataloader
    data_module = PointCloudDataloader()
    data_module.setup("validation")
    model.data_module = data_module
    callbacks = [ModelCheckpoint(monitor="w1p", save_top_k=3, mode="min",filename="{epoch}-{w1p:.5f}",every_n_epochs=1,)]
    # the sets up the model, with some options that can be set
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        logger=logger,
        log_every_n_steps=100,
        max_epochs=20000,
        callbacks=callbacks,
        # progress_bar_refresh_rate=0,
        val_check_interval=1000,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=1,
        enable_progress_bar=False,
        # reload_dataloaders_every_n_epochs=1,
        # fast_dev_run=False,
        # track_grad_norm=1,
        default_root_dir="/beegfs/desy/user/kaechben/calochallenge",
        # reload_dataloaders_every_n_epochs=0,#,config["val_check"] if not config["smart_batching"] else 0,
        #profiler="pytorch"
    )
    print(trainer.default_root_dir)
    # This calls the fit function which trains the model
    print("This is run: ", logger.experiment.name)
    trainer.fit(model,datamodule=data_module)


if __name__ == "__main__":


    config = {
        "batch_size": 1024,
        "part_increase": 10,
        "dropout_gen": 0,
        "dropout": 0.1,
        "gan": "ls",
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
        "name": "CaloChallenge",
        "num_layers_gen": 8,
        "num_layers": 2,
        "opt": "Adam",
        "checkpoint":True,
        "mean_field_loss":True,
        "act": "leaky",
        "stop_mean":True,
        "mass":False,
        "ckpt":None,
        "new_model":True,
        "freq":1,
    }
    #set up WandB logger
    logger = WandbLogger(
        save_dir="/beegfs/desy/user/kaechben/calochallenge",
        sync_tensorboard=False,
        project="CaloChallenge",
    )
    #best function that exists in ml
    logger.experiment.log_code(".")
    # update config with hyperparameters from sweep
    if len(logger.experiment.config.keys()) > 0:
        config.update(**logger.experiment.config)
    print(logger.experiment.dir)
    print("config:", config)
    #ckpt="/beegfs/desy/user/kaechben/pf_t/linear/mao7f3bq/checkpoints/epoch=5513-w1m=0.00020.ckpt"
    ckpt=None
    train(config, ckpt=ckpt,logger=logger)  # load_ckpt=ckptroot=root,
