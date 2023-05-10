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
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, E):
        assert len(data) == len(E), "The lengths of data and E are not equal"
        self.data = data
        self.E = E

    def __getitem__(self, index):
        return self.data[index], self.E[index]

    def __len__(self):
        return len(self.data)
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

def pad_collate_fn_scaled(batch):
    batch,E=zip(*batch)
    max_len = max(len(sample) for sample in batch)
    padded_batch =pad_sequence(batch, batch_first=True, padding_value=0.0)[:,:,:4].float()
    mask = ~(torch.arange(max_len).expand(len(batch), max_len) < torch.tensor([len(sample) for sample in batch]).unsqueeze(1))
    E=torch.from_numpy(np.array(E)).float().reshape(-1)
    return padded_batch,mask,E

def pad_collate_fn(batch):
    batch,E=zip(*batch)
    max_len = max(len(sample) for sample in batch)
    padded_batch =pad_sequence(batch, batch_first=True, padding_value=0.0)[:,:,:4].float()
    mask = ~(torch.arange(max_len).expand(len(batch), max_len) < torch.tensor([len(sample) for sample in batch]).unsqueeze(1))
    E=torch.stack(E).log()-10
    return padded_batch,mask,E

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

    def __init__(self,name,batch_size,max,scaled):
        self.name=name
        self.batch_size=batch_size
        self.max=max
        self.scaled=scaled
        super().__init__()

    def setup(self, stage ):
        # This just sets up the dataloader, nothing particularly important. it reads in a csv, calculates mass and reads out the number particles per jet
        # And adds it to the dataset as variable. The only important thing is that we add noise to zero padded jets
        self.data=torch.load("/beegfs/desy/user/kaechben/calochallenge/pc_{}train_{}.pt".format("scaled_" if self.scaled else "",self.name))
        self.E=self.data["Egen"]
        self.data=self.data["E_z_alpha_r"]
        self.val_data=torch.load("/beegfs/desy/user/kaechben/calochallenge/pc_{}test_{}.pt".format("scaled_" if self.scaled else "",self.name))
        self.val_E=self.val_data["Egen"]
        self.val_data=self.val_data["E_z_alpha_r"]
        self.scaler= ScalerBase(
            transfs=[
                PowerTransformer(method="box-cox", standardize=True),
                Pipeline([('dequantization', DQ()),('minmax_scaler', MinMaxScaler(feature_range=(1e-5, 1-1e-5))),('logit_transformer', LogitTransformer()),("standard_scaler",StandardScaler())]),
                Pipeline([('dequantization', DQ()),('minmax_scaler', MinMaxScaler(feature_range=(1e-5, 1-1e-5))),('logit_transformer', LogitTransformer()),("standard_scaler",StandardScaler())]),
                Pipeline([('dequantization', DQ()),('minmax_scaler', MinMaxScaler(feature_range=(1e-5, 1-1e-5))),('logit_transformer', LogitTransformer()),("standard_scaler",StandardScaler())])]
                ,
            featurenames=["E", "z", "alpha", "r"],
            name=self.name
        )
        if self.max:
            self.train_iterator = BucketBatchSamplerMax(
                                self.data,
                                batch_size = self.batch_size,
                                drop_last=True,
                                max_tokens_per_batch=400000,
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
        self.train_dl = DataLoader(CustomDataset(self.data,self.E), batch_sampler=self.train_iterator, collate_fn=pad_collate_fn if not self.scaled else pad_collate_fn_scaled,num_workers=16)
        self.val_dl = DataLoader(CustomDataset(self.val_data,self.val_E), batch_sampler=self.val_iterator,collate_fn=pad_collate_fn if not self.scaled else pad_collate_fn_scaled,num_workers=16)


    def train_dataloader(self):
        return self.train_dl# DataLoader(self.data, batch_size=10, shuffle=False, num_workers=1, drop_last=False,collate_fn=point_cloud_collate_fn)

    def val_dataloader(self):
        return self.val_dl

if __name__=="__main__":

    loader=PointCloudDataloader("big",64,max=False,scaled=True)
    loader.setup("train")

    for i in loader.val_dataloader():
        assert (i[0]==i[0]).all()
    for i in loader.train_dataloader():
        assert (i[0]==i[0]).all()
