
import os
import numpy as np
import torch
import h5py
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler, QuantileTransformer
from tqdm import tqdm

import matplotlib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler
from scipy.stats import rv_continuous
# Custom transformer for logit transformation
matplotlib.use('Agg')
class LinearInterpolatedDistribution(rv_continuous):
    def __init__(self, x0, x1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x0 = x0
        self.x1 = x1
        self.s=self.x1-self.x0
        self.a = 0
        self.b = 1
        self._compute_normalization_constant()

    def _pdf(self, x):
        return 1/self.k*(self.x0 + self.s * x)

    def _compute_normalization_constant(self):
        self.k = (self.x1 + self.x0)*0.50

    def _cdf(self, x):
        return 1/self.k * ( self.x0 * x + 0.5 * (self.s) * x**2)

    def _ppf(self, q):

        a = 0.5*(self.s)
        b = self.x0
        c= -self.k*q
        discriminant = np.sqrt(b**2 - 4*a*c)
        root1 = ( discriminant - b) / (2*a+1e-5)
        root2 = (- discriminant - b) / (2*a)
        # Use the root that falls within [0, 1]
        #results=np.where((root1 >= 0) & (root1 <= 1), root1, root2)
        return root1 #if (root1 >= 0 and root1 <= 1) else root2

    def rvs(self, size=None):
        u = np.random.uniform(size=size)
        return self._ppf(u)
def tensors_to_point_clouds(l, n):
    cumsum = torch.cat([torch.tensor([0]), torch.from_numpy(n).cumsum(dim=0)])
    l=torch.from_numpy(l)
    point_clouds = [l[cumsum[i]:cumsum[i+1]].tolist() for i in range(len(n))]
    return point_clouds
# creating instance of HighLevelFeatures class to handle geometry based on binning file
class ScalerBase:
    def __init__(self, transfs,name, featurenames,overwrite=False,data_dir="/beegfs/desy/user/kaechben/calochallenge/"):
        self.transfs = transfs
        self.featurenames = featurenames
        self.n_features=len(featurenames)
        self.data_dir=data_dir
        self.scalerpath = Path(data_dir) / "scaler_{}.gz".format(name)
        self.name=name
        if self.scalerpath.is_file() and not overwrite:
             self.transfs = joblib.load(self.scalerpath)

    def save_scalar(self, pcs,save=False):
        # The features need to be converted to numpy immediatly
        # otherwise the queuflow afterwards doesnt work
        self.plot_scaling(pcs)
        pcs = np.hstack(
            [
                self.transfs[0].fit_transform(pcs[:,:1]),
                self.transfs[1].fit_transform(pcs[:,1:])

            ]
        )
        self.plot_scaling(pcs, True)
        pcs=np.hstack(
            [
                self.transfs[0].inverse_transform(pcs[:,:1]),
                self.transfs[1].inverse_transform(pcs[:,1:])

            ])
        self.plot_scaling(pcs, False,True)
        if save:
            joblib.dump(self.transfs, self.scalerpath)
    def transform(self, pcs: np.ndarray):
        assert len(pcs.shape) == 2
        return  np.hstack(
            [
                self.transfs[0].transform(pcs[:,:1]),
                self.transfs[1].transform(pcs[:,1:])

            ]
        )

    def inverse_transform(self, pcs: torch.Tensor):

        orgshape = pcs.shape
        dev = pcs.device
        pcs = pcs.to("cpu").detach().reshape(-1, self.n_features).numpy()
        t_stacked =  np.hstack(
            [
                self.transfs[0].inverse_transform(pcs[:,:1]),
                self.transfs[1].inverse_transform(pcs[:,1:])

            ]
        )
        return torch.from_numpy(t_stacked.reshape(*orgshape)).float().to(dev)

    def plot_scaling(self, pcs, post=False,re=False):
        fig, ax = plt.subplots(1,len(self.featurenames), figsize=(20, 5))
        i=0
        for k, v in zip(self.featurenames, pcs.T):
            bins=min(500,len(np.unique(v)))
            ax[i].hist(v, bins=bins)
            i+=1
            if post:
                savename=self.data_dir+f"{self.name}_post.png"
            elif not re:
                 savename=self.data_dir+ f"{self.name}_pre.png"
            else:
                savename= self.data_dir+f"{self.name}_id.png"
        fig.savefig(savename)

        plt.close(fig)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import numpy as np
from scipy import interpolate
class FirstFeaturePowerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.transformer = PowerTransformer()

    def fit(self, X, y=None):
        self.transformer.fit(X[:, 0:1])
        return self

    def transform(self, X, y=None):
        X[:, 0:1] = self.transformer.transform(X[:, 0:1])
        return X

    def inverse_transform(self, X, y=None):
        X[:, 0:1] = self.transformer.inverse_transform(X[:, 0:1])
        return X
import time

class DQKDE(BaseEstimator, TransformerMixin):
    def __init__(self,name ):
        self.name=name
        pass

    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        fig,ax= plt.subplots(3,2,figsize=(10,5))
        names=["x","y","z"]
        for i in [0,1,2]:  # or replace 3 with X.shape[1] if number of features varies
            data = X[:, i]
            print(names[i])
            ni,_,_=ax[i,0].hist(data, bins=100)
            if i!=1:

                unique_values, counts = np.unique(data, return_counts=True)

                for j in range (len(unique_values)):
                    # Select data points between this value and the next
                    value,count = unique_values[j],counts[j]
                    if j < len(unique_values)-1:
                        nvalue,ncounts = unique_values[j+1],counts[j+1]

                        lid=LinearInterpolatedDistribution(x0=count, x1=ncounts)

                    mask = (X[:,i] >= value) & (X[:,i] < value + 1)
                    samples=lid.rvs(sum(mask))
                    data=X[mask,i]+samples
                    ax[i,1].hist(data, bins=100)
                    X[mask,i]=data
            else:
                X[:, i]=X[:, i]+np.random.rand(*X[:, i].shape)
                ax[i,1].hist(data, bins=100)

            ax[i,0].set_xlabel(names[i])
            ax[i,1].set_xlabel(names[i])
            ax[i,0].set_ylabel("Counts")
            ax[i,1].set_ylabel("Counts")
            nf,_,_=ax[i,0].hist(np.floor(X[:,i]), bins=100, histtype="step",color="red")
            print("difference initial and final",ni-nf)
        plt.tight_layout()
        plt.savefig(f"{self.name}_DQ.png")
        plt.close()
        return X

    def inverse_transform(self, X, y=None):
        X[:, 0:] = np.floor(X[:, 0:])
        X[:, 1:] = np.floor(X[:, 1:])
        X[:, 2:] = np.floor(X[:, 2:])
        return X

class DQ(BaseEstimator, TransformerMixin):
    def __init__(self,name ):
        self.name=name

    def fit(self, X, y=None):
        self.splines=[]
        for i in range(3):
            if i==1:
                self.splines.append(None)
            else:
                self.splines.append(marginal_flows(X[:,i])[1])
        return self

    def transform(self, X, y=None):
        fig,ax= plt.subplots(3,2,figsize=(10,15))
        names=["x","y","z"]
        for i in [0,1,2]:  # or replace 3 with X.shape[1] if number of features varies
            ni,_,_=ax[i,0].hist(X[:,i], bins=100)
            if i!=1:
                kdesample=self.splines[i](np.random.rand(len(X[:,i])*2))
                unique_values = np.unique(X[:,i])

                for value in range(len(unique_values)):
                    mask = (X[:,i] >= value) & (X[:,i] < value + 1)
                    mask_samples = (kdesample >= value) & (kdesample < value + 1)
                    noise= np.random.choice(kdesample[mask_samples],size=sum(mask))
                    X[mask,i]=noise
                    ax[i,1].hist(noise, bins=100)
            else:
                X[:, i]=X[:, i]+np.random.rand(*X[:, i].shape)
                ax[i,1].hist(X[:, i], bins=100)

            nf,_,_=ax[i,0].hist(np.floor(X[:,i]), bins=100, histtype="step",color="red")
            ax[i,0].set_xlabel(names[i])
            ax[i,1].set_xlabel(names[i])
            ax[i,0].set_ylabel("Counts")
            ax[i,1].set_ylabel("Counts")
            print("difference initial and final",ni-nf)
        plt.tight_layout(pad=1.1)
        plt.savefig(f"{self.name}_"+"DQ_kde.png")
        plt.close()
        return X

    def inverse_transform(self, X, y=None):
        X[:, 0:] = np.floor(X[:, 0:])
        X[:, 1:] = np.floor(X[:, 1:])
        X[:, 2:] = np.floor(X[:, 2:])
        return X

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    def fit_transform(self, X, y=None):
        return self.transform(X)
    def transform(self, X, y=None):
        X[:,2]=(X[:,2])**0.5
        return X
    def inverse_transform(self, X, y=None):
        X[:,2]=(X[:,2])**2
        return X
# Custom transformer for inverse logit transformation

class Cart(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_alpha=16
        pass

    def fit(self, X, y=None):
        return self
    def fit_transform(self, X, y=None):
        return self.transform(X)
    def transform(self, X, y=None):
        x=X[:,2]*np.cos(X[:,1]/16*(2*np.pi))
        y=X[:,2]*np.sin(X[:,1]/16*(2*np.pi))
        X[:,2]=X[:,0]
        X[:,1]=y
        X[:,0]=x
        return X
    def inverse_transform(self, X, y=None):
        a=(np.arctan2(X[:,1],X[:,0])+np.pi)*16/(2*np.pi)
        R=np.sqrt(X[:,0]**2+X[:,1]**2)
        X[:,0]=X[:,2]
        X[:,1]=a
        X[:,2]=R
        return X
import scipy
import scipy.stats
def calculate_kde_bandwidth(data):
    N = len(data)  # Number of samples
    sigma = scipy.stats.tstd(data)  # Standard deviation of the samples
    return (4 * (sigma ** 5) / (3 * N)) ** 0.2


from scipy import interpolate
def F(x): #in: 1d array, out: functions transforming array to gauss


        x= np.sort(x)
        x=x+ 0.01 * np.linspace(0,0.01,len(x))
        y=np.linspace(0,1,len(x))
        print(max(np.unique(x,return_counts=True)[1]))


        fun=interpolate.PchipInterpolator(x,y)
        funinv=interpolate.PchipInterpolator(y,x)
        return fun,funinv
def marginal_flows(data):

        f,ffi=F(data)
        fi=lambda x: fbar(ffi,x,min(x),max(x))
        return f,fi
def fbar(f,x,minx,maxx):
        xbar=f(x)+0
        xbar[x<minx]=f(x[x<minx])*np.exp(-abs(x[x<minx]-minx))
        xbar[x>maxx]=f(x[x>maxx])*np.exp(-abs(maxx-x[x>maxx]))
        return xbar

class DQKDEactual(BaseEstimator, TransformerMixin):
    def __init__(self,name ):
        self.name=name
        pass

    def fit(self, X, y=None):
        self.kde=[]
        for i in [0,1,2]:  # or replace 3 with X.shape[1] if number of features varies
            if i!=1:
                k=1000000
                data =  np.random.choice(X[:, i],size=k).reshape(-1,1)+np.random.rand(k,1)
                grid=KernelDensity(kernel='gaussian',bandwidth=calculate_kde_bandwidth(data)[0]).fit(data)
                grid.fit(data)
                _,b,_=plt.hist(data,bins=100)
                plt.hist(grid.sample(k),bins=b)
                plt.savefig(f"{self.name}_kde_{i}.png")
                plt.close()
                self.kde.append(grid)
                # self.kde.append(KernelDensity(kernel='gaussian', bandwidth=).fit(data))
            else:
                self.kde.append(None)

        return self


    def transform(self, X, y=None):
        fig,ax= plt.subplots(3,2,figsize=(10,15))
        names=["x","y","z"]
        for i in [0,1,2]:  # or replace 3 with X.shape[1] if number of features varies
            ni,_,_=ax[i,0].hist(X[:,i], bins=100)
            if i!=1:
                kdesample=self.kde[i].sample(len(X[:,i])*2)
                unique_values = np.unique(X[:,i])

                for value in range(len(unique_values)):
                    mask = (X[:,i] >= value) & (X[:,i] < value + 1)
                    mask_samples = (kdesample >= value) & (kdesample < value + 1)
                    noise= np.random.choice(kdesample[mask_samples],size=sum(mask))
                    X[mask,i]=noise
                    ax[i,1].hist(noise, bins=100)
            else:
                X[:, i]=X[:, i]+np.random.rand(*X[:, i].shape)
                ax[i,1].hist(X[:, i], bins=100)

            nf,_,_=ax[i,0].hist(np.floor(X[:,i]), bins=100, histtype="step",color="red")
            ax[i,0].set_xlabel(names[i])
            ax[i,1].set_xlabel(names[i])
            ax[i,0].set_ylabel("Counts")
            ax[i,1].set_ylabel("Counts")
            print("difference initial and final",ni-nf)
        plt.tight_layout(pad=1.1)
        plt.savefig(f"{self.name}_"+"DQ_kde.png")
        plt.close()
        return X

    def inverse_transform(self, X, y=None):
        X[:, 0:] = np.floor(X[:, 0:])
        X[:, 1:] = np.floor(X[:, 1:])
        X[:, 2:] = np.floor(X[:, 2:])
        return X


if __name__=="__main__":
        #
    big={"train": ["dataset_3_1.hdf5","dataset_3_2.hdf5"], "test": ["dataset_3_3.hdf5"]}
    middle={"train": ["dataset_2_1.hdf5"], "test": ["dataset_2_2.hdf5"]}
    outL=[]

    mode="train"
    outD={}
    i=0
    middle_dataset=True
    name="middle" if middle_dataset else "big"
    if middle_dataset:
        num_z = 45
        num_alpha = 16
        num_r = 9
        files=middle
    data_dir = "/beegfs/desy/user/kaechben/calochallenge/"
    for mode in ["train","test"]:
        for file in files[mode]:
            electron_file = h5py.File(data_dir + file, "r")
            energies = electron_file["incident_energies"][:]
            showers = electron_file["showers"][:].reshape(-1,num_z,num_alpha,num_r)
            energies_layer=showers.sum(-1).sum(-1)

        coords = np.argwhere(showers > 0.) # get indices of non-zero values (shower_id, r, alpha, z)
        vals = showers[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] # get non-zero values
        _, nnz = np.unique(coords[:, 0], return_counts=True) # get number of non-zero values per shower
        coords = coords[:, 1:] # remove shower_id from coords
        start_index = np.zeros(nnz.shape, dtype=np.int64) # create start_index array
        start_index[1:] = np.cumsum(nnz)[:-1] # calculate start_index
        kde=False
        name=name+"_cart"
        if kde=="splines":
            name=name+"_splines"
        # if kde:
        #     name=name+"_kde_small"

        if mode=="train":
            scalar = ScalerBase(
                    transfs=[
                        PowerTransformer(method="box-cox", standardize=True),
                        Pipeline([('dequantization', DQ(name) if kde=="splines" else DQKDEactual(name) if kde else DQKDE(name)),("log_R",LogTransformer()),("cart",Cart()),("standard",StandardScaler())])]#
                        ,#(n_quantiles=100, output_distribution="normal")
                    featurenames=["E", "x", "y", "z"],
                    name=name,
                    overwrite=True
            )
            import copy
            x=np.hstack((vals[:, None], coords))
            scalar.save_scalar(x[:],save=True)
        data=scalar.transform(np.hstack((vals[:, None], coords)))
        #data=tensors_to_point_clouds(data, nnz)
        torch.save({"data":torch.from_numpy(data),"energies":torch.from_numpy(energies),"energies_layer":torch.from_numpy(energies_layer),"n":torch.from_numpy(nnz)}, f"{data_dir}pc_{mode}_{name}.pt",pickle_protocol=4)
        print(f"{data_dir}pc_{mode}_{name}.pt")