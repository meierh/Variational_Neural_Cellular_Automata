import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

import torch
from torch import nn
from torchvision import transforms, datasets
from modules.dml import DiscretizedMixtureLogitsDistribution
from modules.residual import Residual
from modules.vnca import VNCA
from train import train

if __name__ == "__main__":
    z_size = 256
    nca_hid = 128
    n_mixtures = 1
    batch_size = 32
    dmg_size = 16
    p_update = 1.0
    min_steps, max_steps = 64, 128

    filter_size = 5
    pad = filter_size // 2
    encoder_hid = 32
    h = w = 32  # MNIST 图片大小是 28x28, 但是 VNCA 需要输入的图片大小是 32x32
    n_channels = 1  # MNIST 是单通道灰度图像

    def state_to_dist(state):
        return DiscretizedMixtureLogitsDistribution(n_mixtures, state[:, :n_mixtures * 10, :, :])

    encoder = nn.Sequential(
        nn.Conv2d(n_channels, encoder_hid * 2 ** 0, filter_size, padding=pad), nn.ELU(),  # (bs, 32, h, w)
        nn.Conv2d(encoder_hid * 2 ** 0, encoder_hid * 2 ** 1, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 64, h//2, w//2)
        nn.Conv2d(encoder_hid * 2 ** 1, encoder_hid * 2 ** 2, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 128, h//4, w//4)
        nn.Conv2d(encoder_hid * 2 ** 2, encoder_hid * 2 ** 3, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 256, h//8, w//8)
        nn.Conv2d(encoder_hid * 2 ** 3, encoder_hid * 2 ** 4, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 512, h//16, w//16),
        nn.Flatten(),  # (bs, 512*h//16*w//16)
        nn.Linear(encoder_hid * (2 ** 4) * h // 16 * w // 16, 2 * z_size),
    )

    update_net = nn.Sequential(
        nn.Conv2d(z_size, nca_hid, 3, padding=1),
        Residual(
            nn.Conv2d(nca_hid, nca_hid, 1),
            nn.ELU(),
            nn.Conv2d(nca_hid, nca_hid, 1),
        ),
        Residual(
            nn.Conv2d(nca_hid, nca_hid, 1),
            nn.ELU(),
            nn.Conv2d(nca_hid, nca_hid, 1),
        ),
        Residual(
            nn.Conv2d(nca_hid, nca_hid, 1),
            nn.ELU(),
            nn.Conv2d(nca_hid, nca_hid, 1),
        ),
        Residual(
            nn.Conv2d(nca_hid, nca_hid, 1),
            nn.ELU(),
            nn.Conv2d(nca_hid, nca_hid, 1),
        ),
        nn.Conv2d(nca_hid, z_size, 1)
    )
    update_net[-1].weight.data.fill_(0.0)
    update_net[-1].bias.data.fill_(0.0)

    data_dir = os.environ.get('DATA_DIR') or "data"
    #modify the transform to resize the image to 32x32
    tp = transforms.Compose([
        transforms.Resize((h, w)),  # 修改图像大小
        transforms.ToTensor()
    ])
    train_data = datasets.MNIST(data_dir, train=True, download=True, transform=tp)
    test_data = datasets.MNIST(data_dir, train=False, download=True, transform=tp)

    # 分割训练数据集和验证数据集
    val_size = 10000
    train_size = len(train_data) - val_size
    train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

    vnca = VNCA(h, w, n_channels, z_size, encoder, update_net, train_data, val_data, test_data, state_to_dist, batch_size, dmg_size, p_update, min_steps, max_steps)
    vnca.eval_batch()
    train(vnca, n_updates=8, eval_interval=4)
    vnca.test(1)