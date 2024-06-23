import os
import urllib

import certifi
import numpy as np
import torch as t
from torch.utils.data import Dataset
from medmnist import PathMNIST
from medmnist import ChestMNIST
from medmnist import DermaMNIST
from medmnist import OCTMNIST
from medmnist import PneumoniaMNIST
from medmnist import RetinaMNIST
from medmnist import BreastMNIST
from medmnist import BloodMNIST
from medmnist import TissueMNIST
from medmnist import OrganAMNIST
from medmnist import OrganCMNIST
from medmnist import OrganSMNIST

class PathMNISTDataset(Dataset):
    splits = {"train","val","test"}

    def __init__(self, splitSet):
        assert splitSet in self.splits
        self.thisSet =  PathMNIST(split=splitSet, download=True)

    def __getitem__(self, index):
        return self.thisSet.__getitem__(index)[0], 0  # placeholder label

    def __len__(self):
        return self.thisSet.__len__()
    
class ChestMNISTDataset(Dataset):
    splits = {"train","val","test"}

    def __init__(self, splitSet):
        assert splitSet in self.splits
        self.thisSet =  ChestMNIST(split=splitSet, download=True)

    def __getitem__(self, index):
        return self.thisSet.__getitem__(index)[0], 0  # placeholder label

    def __len__(self):
        return self.thisSet.__len__()
    
class DermaMNISTDataset(Dataset):
    splits = {"train","val","test"}

    def __init__(self, splitSet):
        assert splitSet in self.splits
        self.thisSet =  DermaMNIST(split=splitSet, download=True)

    def __getitem__(self, index):
        return self.thisSet.__getitem__(index)[0], 0  # placeholder label

    def __len__(self):
        return self.thisSet.__len__()
    
class OCTMNISTDataset(Dataset):
    splits = {"train","val","test"}

    def __init__(self, splitSet):
        assert splitSet in self.splits
        self.thisSet =  OCTMNIST(split=splitSet, download=True)

    def __getitem__(self, index):
        return self.thisSet.__getitem__(index)[0], 0  # placeholder label

    def __len__(self):
        return self.thisSet.__len__()
    
class PneumoniaMNISTDataset(Dataset):
    splits = {"train","val","test"}

    def __init__(self, splitSet):
        assert splitSet in self.splits
        self.thisSet =  PneumoniaMNIST(split=splitSet, download=True)

    def __getitem__(self, index):
        return self.thisSet.__getitem__(index)[0], 0  # placeholder label

    def __len__(self):
        return self.thisSet.__len__()
    
class RetinaMNISTDataset(Dataset):
    splits = {"train","val","test"}

    def __init__(self, splitSet):
        assert splitSet in self.splits
        self.thisSet =  RetinaMNIST(split=splitSet, download=True)

    def __getitem__(self, index):
        return self.thisSet.__getitem__(index)[0], 0  # placeholder label

    def __len__(self):
        return self.thisSet.__len__()

class BreastMNISTDataset(Dataset):
    splits = {"train","val","test"}

    def __init__(self, splitSet):
        assert splitSet in self.splits
        self.thisSet =  BreastMNIST(split=splitSet, download=True)

    def __getitem__(self, index):
        return self.thisSet.__getitem__(index)[0], 0  # placeholder label

    def __len__(self):
        return self.thisSet.__len__()

class BloodMNISTDataset(Dataset):
    splits = {"train","val","test"}

    def __init__(self, splitSet):
        assert splitSet in self.splits
        self.thisSet =  BloodMNIST(split=splitSet, download=True)

    def __getitem__(self, index):
        return self.thisSet.__getitem__(index)[0], 0  # placeholder label

    def __len__(self):
        return self.thisSet.__len__()
    
class TissueMNISTDataset(Dataset):
    splits = {"train","val","test"}

    def __init__(self, splitSet):
        assert splitSet in self.splits
        self.thisSet =  TissueMNIST(split=splitSet, download=True)

    def __getitem__(self, index):
        return self.thisSet.__getitem__(index)[0], 0  # placeholder label

    def __len__(self):
        return self.thisSet.__len__()
    
class OrganAMNISTDataset(Dataset):
    splits = {"train","val","test"}

    def __init__(self, splitSet):
        assert splitSet in self.splits
        self.thisSet =  OrganAMNIST(split=splitSet, download=True)

    def __getitem__(self, index):
        return self.thisSet.__getitem__(index)[0], 0  # placeholder label

    def __len__(self):
        return self.thisSet.__len__()
    
class OrganCMNISTDataset(Dataset):
    splits = {"train","val","test"}

    def __init__(self, splitSet):
        assert splitSet in self.splits
        self.thisSet =  OrganCMNIST(split=splitSet, download=True)

    def __getitem__(self, index):
        return self.thisSet.__getitem__(index)[0], 0  # placeholder label

    def __len__(self):
        return self.thisSet.__len__()
    
class OrganSMNISTDataset(Dataset):
    splits = {"train","val","test"}

    def __init__(self, splitSet):
        assert splitSet in self.splits
        self.thisSet =  OrganSMNIST(split=splitSet, download=True)

    def __getitem__(self, index):
        return self.thisSet.__getitem__(index)[0], 0  # placeholder label

    def __len__(self):
        return self.thisSet.__len__()
    
AllDatasets = {
    PathMNISTDataset,
    ChestMNISTDataset,
    DermaMNISTDataset,
    OCTMNISTDataset,
    PneumoniaMNISTDataset,
    RetinaMNISTDataset,
    BreastMNISTDataset,
    BloodMNISTDataset,
    TissueMNISTDataset,
    OrganAMNISTDataset,
    OrganCMNISTDataset,
    OrganSMNISTDataset
}