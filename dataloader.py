import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, BatchSampler
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
from helpers import ScalerBase,LogitTransformer,DQ
from sklearn.preprocessing import StandardScaler, PowerTransformer,MinMaxScaler
import copy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler
class BucketBatchSampler(BatchSampler):
    def __init__(self, data_source, batch_size, shuffle=True, drop_last=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        # Sort sequences by length
        indices = sorted(indices, key=lambda x: len(self.data_source[x]))
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
    max_len = max(len(sample) for sample in batch)
    padded_batch =pad_sequence(batch, batch_first=True, padding_value=0.0)[:,:,:4].float()
    mask = ~(torch.arange(max_len).expand(len(batch), max_len) < torch.tensor([len(sample) for sample in batch]).unsqueeze(1))
    return padded_batch,mask

# Pad the sequences using pad_sequence()

class PointCloudDataloader(pl.LightningDataModule):
    """This is more or less standard boilerplate coded that builds the data loader of the training
    one thing to note is the custom standard scaler that works on tensors
   """

    def __init__(self):
        super().__init__()

    def setup(self, stage ,n=None ):
        # This just sets up the dataloader, nothing particularly important. it reads in a csv, calculates mass and reads out the number particles per jet
        # And adds it to the dataset as variable. The only important thing is that we add noise to zero padded jets
        self.data=torch.load("/beegfs/desy/user/kaechben/calochallenge/pc_train.pt")["E_z_alpha_r"]
        self.val_data=torch.load("/beegfs/desy/user/kaechben/calochallenge/pc_test.pt")["E_z_alpha_r"]
        self.scaler= ScalerBase(
            transfs=[
                PowerTransformer(method="box-cox", standardize=True),
                Pipeline([('dequantization', DQ()),('minmax_scaler', MinMaxScaler(feature_range=(1e-5, 1-1e-5))),('logit_transformer', LogitTransformer()),("standard_scaler",StandardScaler())]),
                Pipeline([('dequantization', DQ()),('minmax_scaler', MinMaxScaler(feature_range=(1e-5, 1-1e-5))),('logit_transformer', LogitTransformer()),("standard_scaler",StandardScaler())]),
                Pipeline([('dequantization', DQ()),('minmax_scaler', MinMaxScaler(feature_range=(1e-5, 1-1e-5))),('logit_transformer', LogitTransformer()),("standard_scaler",StandardScaler())])]
                ,
            featurenames=["E", "z", "alpha", "r"],
        )

        self.train_iterator = BucketBatchSampler(
                            self.data,
                            batch_size = 64,
                            drop_last=True,
                            shuffle=True
                            )
        self.val_iterator = BucketBatchSampler(
                            self.val_data,
                            batch_size = 10000,
                            drop_last=True,
                            shuffle=True
                            )
        self.train_dl = DataLoader(self.data, batch_sampler=self.train_iterator, collate_fn=pad_collate_fn,num_workers=40)
        self.val_dl = DataLoader(self.val_data, batch_sampler=self.val_iterator ,collate_fn=pad_collate_fn,num_workers=40)


    def train_dataloader(self):
        return self.train_dl# DataLoader(self.data, batch_size=10, shuffle=False, num_workers=1, drop_last=False,collate_fn=point_cloud_collate_fn)

    def val_dataloader(self):
        return self.val_dl

if __name__=="__main__":

    loader=PointCloudDataloader()
    loader.setup("train")

    for i in loader.val_dataloader():
        print(i)
        break