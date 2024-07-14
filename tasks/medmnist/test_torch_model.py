import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import PathMNISTDataset, DermaMNISTDataset, RetinaMNISTDataset, BloodMNISTDataset, BreastMNISTDataset

# 确认导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

from modules.dml import DiscretizedMixtureLogitsDistribution
from modules.loss import elbo, iwae
from modules.vnca import VNCA
from modules.residual import Residual

# 设置一些参数
z_size = 256
nca_hid = 128
filter_size = 5
pad = filter_size // 2
encoder_hid = 32
h = w = 32
n_channels = 3
batch_size = 32
n_mixtures = 1

# 定义state_to_dist函数
def state_to_dist(state):
    return DiscretizedMixtureLogitsDistribution(n_mixtures, state[:, :n_mixtures * 10, :, :])

# 定义encoder
encoder = nn.Sequential(
    nn.Conv2d(n_channels, encoder_hid * 2 ** 0, filter_size, padding=pad), nn.ELU(),
    nn.Conv2d(encoder_hid * 2 ** 0, encoder_hid * 2 ** 1, filter_size, padding=pad, stride=2), nn.ELU(),
    nn.Conv2d(encoder_hid * 2 ** 1, encoder_hid * 2 ** 2, filter_size, padding=pad, stride=2), nn.ELU(),
    nn.Conv2d(encoder_hid * 2 ** 2, encoder_hid * 2 ** 3, filter_size, padding=pad, stride=2), nn.ELU(),
    nn.Conv2d(encoder_hid * 2 ** 3, encoder_hid * 2 ** 4, filter_size, padding=pad, stride=2), nn.ELU(),
    nn.Flatten(),
    nn.Linear(encoder_hid * (2 ** 4) * h // 16 * w // 16, 2 * z_size),
)

# 定义update_net
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

# 定义数据集和数据加载器
data_dir = os.environ.get('DATA_DIR', "data")
dataset = BloodMNISTDataset(splitSet='test')
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 初始化模型
vnca = VNCA(h, w, n_channels, z_size, encoder, update_net, None, None, None, state_to_dist, batch_size, 16, 1.0, 64, 128, elbo, elbo, 1e-4)

# 加载模型权重
results_dir = os.path.join(grandparent_dir, 'results')
model_path = os.path.join(results_dir, f"vnca_model_bloodmnist_50000_1000.pth")
vnca.load_state_dict(torch.load(model_path))
vnca.eval()

# 计算损失
total_loss = 0
criterion = elbo  # 使用你选择的损失函数
n_samples = 2  # 你可以根据需要调整此值

with torch.no_grad():
    for images, labels in data_loader:
        images = images.to(vnca.device)
        labels = labels.to(vnca.device)
        loss, _, _, _, _, _ = vnca(images, n_samples=n_samples, loss_fn=criterion)
        total_loss += loss.mean().item()

average_loss = total_loss / len(data_loader)
print(f'Average loss: {average_loss}')
