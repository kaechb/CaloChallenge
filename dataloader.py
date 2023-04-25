import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, BatchSampler
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
from helpers import ScalerBase
from sklearn.preprocessing import StandardScaler, PowerTransformer
class PointCloudDataset(Dataset):
    def __init__(self, path: str,scale=False):
        self.data = np.load(path)

        self.points = torch.from_numpy(self.data["points"])
        # self.e_in_hat = torch.from_numpy(self.data["e_in_hat"])
        # self.e_sum_hat = torch.from_numpy(self.data["e_sum_hat"])
        # self.n_hat = torch.from_numpy(self.data["n_hat"])
        self.start_index = torch.from_numpy(self.data["start_index"])
        self.nnz = torch.from_numpy(self.data["nnz"])
        # self.e_in = torch.from_numpy(self.data["e_in"])
        nmax=max(self.nnz)
        #self.mask = ~(torch.arange(nmax).expand(len(self.nnz), nmax) < self.nnz.unsqueeze(1))
        sequences = []

        flat=torch.tensor(0,4)
        for start_index,length in zip(self.start_index,self.nnz):
            end_idx =  start_index+ length
            sequence = self.points[start_index:end_idx]
            sequences.append(sequence)





        self.sequences = sequences
        #self.points = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True).reshape(-1,nmax,4)


    # def __len__(self):
    #     return len(self.start_index)

    # def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
    #     start = self.start_index[idx]
    #     end = start + self.nnz[idx]
    #     return self.points[start:end, :], self.nnz[idx], self.e_in[idx], self.n_hat[idx], self.e_in_hat[idx], self.e_sum_hat[idx]
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
    padded_batch =pad_sequence(batch, batch_first=True, padding_value=0.0)[:,:,:4]
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
        self.data=torch.load("/beegfs/desy/user/mscham/training_data/calochallange2/pc_train.pt")["E_z_alpha_r_zp_eta_phi"]
        self.val_data=torch.load("/beegfs/desy/user/mscham/training_data/calochallange2/pc_test.pt")["E_z_alpha_r_zp_eta_phi"]
        self.scaler = ScalerBase(
            transfs=[
                PowerTransformer(method="box-cox", standardize=True),
                StandardScaler(),
                StandardScaler(),
                StandardScaler(),
                StandardScaler(),
                StandardScaler(),
                StandardScaler(),
            ],
            featurenames=["E", "z", "alpha", "r", "zp", "eta", "phi"],
        )

        self.train_iterator = BucketBatchSampler(
                            self.data,
                            batch_size = 64,
                            drop_last=True,
                            shuffle=True
                            )
        self.train_dl = DataLoader(self.data, batch_sampler=self.train_iterator, collate_fn=pad_collate_fn,num_workers=1)



    def train_dataloader(self):
        return self.train_dl# DataLoader(self.data, batch_size=10, shuffle=False, num_workers=1, drop_last=False,collate_fn=point_cloud_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_data,batch_size=len(self.val_data),collate_fn=pad_collate_fn)

if __name__=="__main__":

    loader=PointCloudDataloader()
    loader.setup("train")

    for i in loader.val_dataloader():
        print(i)
        break