import os
# modify
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

import distutils

from torch import nn, t
from torch.distributions import Bernoulli
from torch.utils.data import ConcatDataset

from modules.vnca import VNCA
from tasks.medmnist.data import TissueMNISTDataset
from tasks.medmnist.data import AllDatasets
from train import train

def state_to_dist(state):
    return Bernoulli(logits=state[:, :1, :, :])


if __name__ == "__main__":
    for dataset in AllDatasets:
        oneSet = dataset("train")
        dataitem = oneSet.__getitem__(0)
        image = dataitem[0]
        print("image.size:", image.size)
    
    z_size = 128
    nca_hid = 128
    batch_size = 128
    dmg_size = 14

    filter_size = 5
    pad = filter_size // 2
    encoder_hid = 32
    h = w = 28
    n_channels = 1
    
    encoder = nn.Sequential(
        nn.Conv2d(n_channels, encoder_hid * 2 ** 0, filter_size, padding=pad + 2), nn.ELU(),  # (bs, 32, h, w)
        nn.Conv2d(encoder_hid * 2 ** 0, encoder_hid * 2 ** 1, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 64, h//2, w//2)
        nn.Conv2d(encoder_hid * 2 ** 1, encoder_hid * 2 ** 2, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 128, h//4, w//4)
        nn.Conv2d(encoder_hid * 2 ** 2, encoder_hid * 2 ** 3, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 256, h//8, w//8)
        nn.Conv2d(encoder_hid * 2 ** 3, encoder_hid * 2 ** 4, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 512, h//16, w//16),
        nn.Flatten(),  # (bs, 512*h//16*w//16)
        nn.Linear(encoder_hid * (2 ** 4) * 2 * 2, 2 * z_size),
    )

    update_net = nn.Sequential(
        nn.Conv2d(z_size, nca_hid, 3, padding=1),
        nn.ELU(),
        nn.Conv2d(nca_hid, z_size, 1, bias=False)
    )
    update_net[-1].weight.data.fill_(0.0)
    
    train_data, val_data, test_data = TissueMNISTDataset("train"), TissueMNISTDataset("val"), TissueMNISTDataset("test")
    train_data = ConcatDataset((train_data, val_data))
    
    print("len(train_data):", len(train_data))
    dataitem = train_data.__getitem__(0)
    print("type(dataitem[0]):", type(dataitem[0]))
    print("type(dataitem[1]):", type(dataitem[1]))
    image = dataitem[0]
    print("type(image):", type(image))
    print("image.size:", image.size)

    vnca = VNCA(h, w, n_channels, z_size, encoder, update_net, train_data, test_data, test_data, state_to_dist, batch_size, dmg_size, 1.0, 32, 64)
    vnca.eval_batch()
    #original train(vnca, n_updates=100_000, eval_interval=100)
    # vnca.test(128)
    train(vnca, n_updates=50, eval_interval=5)
    vnca.test(4)