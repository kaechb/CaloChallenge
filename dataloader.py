import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, Dataset, Sampler
import numpy as np
import pytorch_lightning as pl
class PointCloudDataset(Dataset):
    def __init__(self, path: str):
        self.data = np.load(path)

        self.points = torch.from_numpy(self.data["points"])
        self.e_in_hat = torch.from_numpy(self.data["e_in_hat"])
        self.e_sum_hat = torch.from_numpy(self.data["e_sum_hat"])
        self.n_hat = torch.from_numpy(self.data["n_hat"])
        self.start_index = torch.from_numpy(self.data["start_index"])
        self.nnz = torch.from_numpy(self.data["nnz"])
        self.e_in = torch.from_numpy(self.data["e_in"])

    def __len__(self):
        return len(self.start_index)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        start = self.start_index[idx]
        end = start + self.nnz[idx]
        return self.points[start:end, :], self.nnz[idx], self.e_in[idx], self.n_hat[idx], self.e_in_hat[idx], self.e_sum_hat[idx]


def point_cloud_collate_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    points, nnz, e_in, n_hat, e_in_hat, e_sum_hat = zip(*batch)

    points = torch.cat(points, dim=0)
    nnz = torch.stack(nnz)
    e_in = torch.stack(e_in)
    n_hat = torch.stack(n_hat)
    e_in_hat = torch.stack(e_in_hat)
    e_sum_hat = torch.stack(e_sum_hat)
    sequences = []
    start_idx = 0
    for length in nnz:
        end_idx = start_idx + length
        sequence = points[start_idx:end_idx]
        sequences.append(sequence)
        start_idx = end_idx

# Pad the sequences using pad_sequence()
    nmax=max(nnz)
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True).reshape(-1,nmax,4)
    mask = ~(torch.arange(nmax).expand(len(nnz), nmax) < nnz.unsqueeze(1))
    return [padded_sequences, mask]

class PointCloudDataloader(pl.LightningDataModule):
    """This is more or less standard boilerplate coded that builds the data loader of the training
    one thing to note is the custom standard scaler that works on tensors
   """

    def __init__(self):
        super().__init__()

    def setup(self, stage ,n=None ):
        # This just sets up the dataloader, nothing particularly important. it reads in a csv, calculates mass and reads out the number particles per jet
        # And adds it to the dataset as variable. The only important thing is that we add noise to zero padded jets
        self.data=PointCloudDataset("/beegfs/desy/user/schnakes/public/train.npz")
        self.val_data=PointCloudDataset("/beegfs/desy/user/schnakes/public/val.npz")

    def train_dataloader(self):
            return DataLoader(self.data, batch_size=10, shuffle=False, num_workers=1, drop_last=False,collate_fn=point_cloud_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=500000, drop_last=False,num_workers=1,collate_fn=point_cloud_collate_fn)

if __name__=="__main__":

    loader=PointCloudDataloader()
    loader.setup("train")
    for i in loader.train_dataloader():
        print(i)
        break