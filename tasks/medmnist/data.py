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

#modify
from torchvision import transforms
h = w = 32  # 修改图像大小

class PathMNISTDataset(Dataset):
    splits = {"train","val","test"}

    def __init__(self, splitSet):
        assert splitSet in self.splits
        self.thisSet =  PathMNIST(split=splitSet, download=True)
        #modify
        self.transform = transforms.Compose([
            transforms.Resize((h, w)), 
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float())  # 添加这一步，将数据转换为二值
        ])

    def __getitem__(self, index):
        #original return self.thisSet.__getitem__(index)[0], 0  # placeholder label
        image, label = self.thisSet.__getitem__(index)
        image = self.transform(image)  # 转换为张量
        return image, label  # 返回标签

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
            transforms.Lambda(lambda x: (x > 0.5).float())  # 添加这一步，将数据转换为二值
        ])

    def __getitem__(self, index):
        #original return self.thisSet.__getitem__(index)[0], 0  # placeholder label
        image, label = self.thisSet.__getitem__(index)
        image = self.transform(image)  # 转换为张量
        return image, label  # 返回标签

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
            transforms.Lambda(lambda x: (x > 0.5).float())  # 添加这一步，将数据转换为二值
        ])

    def __getitem__(self, index):
        #original return self.thisSet.__getitem__(index)[0], 0  # placeholder label
        image, label = self.thisSet.__getitem__(index)
        image = self.transform(image)  # 转换为张量
        return image, label  # 返回标签

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
            transforms.Lambda(lambda x: (x > 0.5).float())  # 添加这一步，将数据转换为二值
        ])

    def __getitem__(self, index):
        #original return self.thisSet.__getitem__(index)[0], 0  # placeholder label
        image, label = self.thisSet.__getitem__(index)
        image = self.transform(image)  # 转换为张量
        return image, label  # 返回标签

    def __len__(self):
        return self.thisSet.__len__()
    
AllDatasets = {
    PathMNISTDataset,
    DermaMNISTDataset,
    RetinaMNISTDataset,
    BloodMNISTDataset,
}