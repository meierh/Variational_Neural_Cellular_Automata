import os
import urllib

import certifi
import numpy as np
import torch as t
from torch.utils.data import Dataset
from medmnist import PathMNIST
from medmnist import DermaMNIST
from medmnist import RetinaMNIST
from medmnist import BloodMNIST

from medmnist import BreastMNIST


#modify
from torchvision import transforms
h = w = 32  # resize to 32x32

class PathMNISTDataset(Dataset):
    splits = {"train","val","test"}

    def __init__(self, splitSet):
        assert splitSet in self.splits
        self.thisSet =  PathMNIST(split=splitSet, download=True)
        #modify
        self.transform = transforms.Compose([
            transforms.Resize((h, w)), 
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float())
        ])

    def __getitem__(self, index):
        #original return self.thisSet.__getitem__(index)[0], 0  # placeholder label
        image, label = self.thisSet.__getitem__(index)
        image = self.transform(image) 
        return image, label  

    def __len__(self):
        return self.thisSet.__len__()
    
class DermaMNISTDataset(Dataset):
    splits = {"train","val","test"}

    def __init__(self, splitSet):
        assert splitSet in self.splits
        self.thisSet =  DermaMNIST(split=splitSet, download=True)
        #modify
        self.transform = transforms.Compose([
            transforms.Resize((h, w)), 
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float())
        ])

    def __getitem__(self, index):
        #original return self.thisSet.__getitem__(index)[0], 0  # placeholder label
        image, label = self.thisSet.__getitem__(index)
        image = self.transform(image) 
        return image, label  

    def __len__(self):
        return self.thisSet.__len__()
    
class RetinaMNISTDataset(Dataset):
    splits = {"train","val","test"}

    def __init__(self, splitSet):
        assert splitSet in self.splits
        self.thisSet =  RetinaMNIST(split=splitSet, download=True)
        #modify
        self.transform = transforms.Compose([
            transforms.Resize((h, w)), 
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float()) 
        ])

    def __getitem__(self, index):
        #original return self.thisSet.__getitem__(index)[0], 0  # placeholder label
        image, label = self.thisSet.__getitem__(index)
        image = self.transform(image)  
        return image, label 

    def __len__(self):
        return self.thisSet.__len__()

class BloodMNISTDataset(Dataset):
    splits = {"train","val","test"}

    def __init__(self, splitSet):
        assert splitSet in self.splits
        self.thisSet =  BloodMNIST(split=splitSet, download=True)
        #modify
        self.transform = transforms.Compose([
            transforms.Resize((h, w)), 
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float())
        ])

    def __getitem__(self, index):
        #original return self.thisSet.__getitem__(index)[0], 0  # placeholder label
        image, label = self.thisSet.__getitem__(index)
        image = self.transform(image)
        return image, label 

    def __len__(self):
        return self.thisSet.__len__()

class BreastMNISTDataset(Dataset):
    splits = {"train","val","test"}

    def __init__(self, splitSet):
        assert splitSet in self.splits
        self.thisSet =  BreastMNIST(split=splitSet, download=True)
        #modify
        self.transform = transforms.Compose([
            transforms.Resize((h, w)), 
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float())  # change to binary
        ])

    def __getitem__(self, index):
        #original return self.thisSet.__getitem__(index)[0], 0  # placeholder label
        image, label = self.thisSet.__getitem__(index)
        image = self.transform(image)  # turn it to tensor
        image = np.tile(image, (3, 1, 1)) # change grayscale to fake RGB, to solve the problem of input channel
        return image, label

    def __len__(self):
        return self.thisSet.__len__()
    
AllDatasets = {
    PathMNISTDataset,
    DermaMNISTDataset,
    RetinaMNISTDataset,
    BloodMNISTDataset,
    BreastMNISTDataset,
}