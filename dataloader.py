import copy

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, PowerTransformer,
                                   StandardScaler)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import BatchSampler, DataLoader, Dataset

from preprocess import DQ, DQKDE, Cart, LogTransformer, ScalerBase, DQKDEactual,StandardScaler, MinMaxScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data import Dataset

def tensors_to_point_clouds(l, n):
    cumsum = torch.cat([torch.tensor([0]), n.cumsum(dim=0)])
    point_clouds = [l[cumsum[i]:cumsum[i+1]] for i in range(len(cumsum)-1)]
    return point_clouds

class CustomDataset(Dataset):
    #thanks CHAT-GPT
    def __init__(self, data, E,layerE,n):
        self.data = data
        self.E = E
        self.layerE=layerE
        self.length = len(E)

    def __getitem__(self, idx):
        return self.data[idx], self.E[idx], self.layerE[idx]
    def __len__(self):
        return len(self.n)

class BucketBatchSampler(BatchSampler):
    #thanks CHAT-GPT
    def __init__(self, data_source, batch_size, shuffle=True, drop_last=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        # Sort sequences by length
        #indices = sorted(indices, key=lambda x: len(self.data_source[x]))
        # Create batches based on the sorted indices
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        if self.drop_last and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]
        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size


def pad_collate_fn(batch):
    batch,E,layerE=zip(*batch)
    max_len = max(len(sample) for sample in batch)
    padded_batch =pad_sequence(batch, batch_first=True, padding_value=0.0)[:,:,:4].float()
    mask = ~(torch.arange(max_len).expand(len(batch), max_len) < torch.tensor([len(sample) for sample in batch]).unsqueeze(1))
    E=torch.stack(E).log()-10
    return padded_batch,mask,E,layerE

# Pad the sequences using pad_sequence()


class BucketBatchSamplerMax(BatchSampler):
    def __init__(self, data_source, batch_size, max_tokens_per_batch=400000, shuffle=True, drop_last=False):
        self.data_source = data_source

        self.max_tokens_per_batch = max_tokens_per_batch
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.batch_size=batch_size

    def __iter__(self):
        indices = list(range(len(self.data_source)))

        if self.shuffle:
            np.random.shuffle(indices)

        # Sort sequences by length
        indices = sorted(indices, key=lambda x: len(self.data_source[x]))

        # Create batches based on the total number of tokens per batch
        batches = []
        batch = []
        batch_tokens = 0
        for idx in indices:
            sample_len = len(self.data_source[idx])
            if batch_tokens + sample_len > self.max_tokens_per_batch or len(batch) >= self.batch_size:
                if len(batch) > 0:
                    batches.append(batch)
                batch = []
                batch_tokens = 0

            batch.append(idx)
            batch_tokens += sample_len
        if len(batch) > 0 and not self.drop_last:
            batches.append(batch)
        if self.shuffle:
            np.random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size
class PointCloudDataloader(pl.LightningDataModule):
    """This is more or less standard boilerplate coded that builds the data loader of the training
    one thing to note is the custom standard scaler that works on tensors
   """

    def __init__(self,name,batch_size,max,cartesian,scale_E):
        self.name=name
        self.batch_size=batch_size
        self.max=max
        self.cartesian=cartesian
        self.scale_E=scale_E
        if cartesian:
            self.name=self.name+"_cart"
        super().__init__()

    def setup(self, stage ):
        # This just sets up the dataloader, nothing particularly important. it reads in a csv, calculates mass and reads out the number particles per jet
        # And adds it to the dataset as variable. The only important thing is that we add noise to zero padded jets
        self.data=torch.load("/beegfs/desy/user/kaechben/calochallenge/pc_train_middle_cart.pt".format(self.name))
        self.E=self.data["energies"].float()
        self.E_layer=self.data["energies_layer"].float()
        self.n=self.data["n"]

        self.data=self.data["data"].float()
        self.data=tensors_to_point_clouds(self.data,self.n)
        #self.data=tensors_to_point_clouds(self.data,self.n)
        self.val_data=torch.load("/beegfs/desy/user/kaechben/calochallenge/pc_train_middle_cart.pt")
        self.val_E=self.val_data["energies"].float()
        self.val_E_layer=self.val_data["energies_layer"].float()
        self.val_n=self.val_data["n"]
        self.val_data=self.val_data["data"].float()
        self.val_data=tensors_to_point_clouds(self.val_data,self.val_n)
        self.average_n=self.n.float().mean()
        self.scaler = ScalerBase(
            transfs=[
                PowerTransformer(method="box-cox", standardize=True),
                Pipeline([('dequantization', DQKDE(name=self.name)),("log_R",LogTransformer()), ('cartesian', Cart()),("standard",StandardScaler())])]
                    ,#(n_quantiles=100, output_distribution="normal")
                featurenames=["E", "x", "y", "z"],
                name=self.name,
                overwrite=False)

        # E_layer=self.E_layer.reshape(-1,1)
        if self.scale_E:
            E_layer=self.E_layer.reshape(-1,1)
            E_layer[E_layer>0]=torch.from_numpy(self.scaler.transfs[0].transform((E_layer[E_layer>0]).reshape(-1,1))).float().reshape(-1)
            self.E_layer=E_layer.reshape(-1,45)
            E_layer=self.val_E_layer.reshape(-1,1)

            self.val_E_layer[self.val_E_layer>0]=torch.from_numpy(self.scaler.transfs[0].transform(self.val_E_layer[self.val_E_layer>0].reshape(-1,1))).float().reshape(-1)
            self.val_E_layer=self.val_E_layer.reshape(-1,45).float()
            self.E=torch.from_numpy(self.scaler.transfs[0].transform(self.E)).float()
            self.n=self.n
            self.val_E=torch.from_numpy(self.scaler.transfs[0].transform(self.val_E)).float()
            self.val_n=self.val_n
        if self.max:
            self.train_iterator = BucketBatchSamplerMax(
                                self.data,
                                batch_size = self.batch_size,
                                drop_last=True,
                                max_tokens_per_batch=1200000,
                                shuffle=True
                                )
            self.val_iterator = BucketBatchSamplerMax(
                                self.val_data,
                                batch_size = self.batch_size,
                                max_tokens_per_batch=400000,
                                drop_last=False,
                                shuffle=True
                                )
        else:
            self.train_iterator = BucketBatchSampler(
                                self.data,

                                batch_size = self.batch_size//2,
                                drop_last=True,
                                shuffle=True
                                )
            self.val_iterator = BucketBatchSampler(
                                self.val_data,

                                batch_size = self.batch_size,
                                drop_last=False,
                                shuffle=True
                                )
    def train_dataloader(self):
        return DataLoader(CustomDataset(self.data,self.E,self.E_layer,self.n), batch_sampler=self.train_iterator, collate_fn=pad_collate_fn ,num_workers=16)

    def val_dataloader(self):
        return  DataLoader(CustomDataset(self.val_data,self.val_E,self.val_E_layer,self.val_n), batch_sampler=self.val_iterator,collate_fn=pad_collate_fn ,num_workers=16)

if __name__=="__main__":

    loader=PointCloudDataloader("middle",64,max=False,cartesian=True,scale_E=True)
    loader.setup("train")

    for i in loader.val_dataloader():
        assert (i[0]==i[0]).all()
    for i in loader.train_dataloader():
        assert (i[0]==i[0]).all()
