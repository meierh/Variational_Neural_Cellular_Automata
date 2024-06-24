import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from data import PathMNISTDataset, DermaMNISTDataset, RetinaMNISTDataset, BloodMNISTDataset, BreastMNISTDataset
from modules.dml import DiscretizedMixtureLogitsDistribution
from modules.residual import Residual
from modules.vnca import VNCA
from train import train
import torch

selected_dataset = "pathmnist"
pic_channels = 3
n_updates_s = 20
eval_interval_s = 5
num_test = 64

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
    h = w = 32
    n_channels = pic_channels

    def state_to_dist(state):
        return DiscretizedMixtureLogitsDistribution(n_mixtures, state[:, :n_mixtures * 10, :, :])

    encoder = nn.Sequential(
        nn.Conv2d(n_channels, encoder_hid * 2 ** 0, filter_size, padding=pad), nn.ELU(),
        nn.Conv2d(encoder_hid * 2 ** 0, encoder_hid * 2 ** 1, filter_size, padding=pad, stride=2), nn.ELU(),
        nn.Conv2d(encoder_hid * 2 ** 1, encoder_hid * 2 ** 2, filter_size, padding=pad, stride=2), nn.ELU(),
        nn.Conv2d(encoder_hid * 2 ** 2, encoder_hid * 2 ** 3, filter_size, padding=pad, stride=2), nn.ELU(),
        nn.Conv2d(encoder_hid * 2 ** 3, encoder_hid * 2 ** 4, filter_size, padding=pad, stride=2), nn.ELU(),
        nn.Flatten(),
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

    data_dir = os.environ.get('DATA_DIR', "data")
    dataset_classes = {
        "pathmnist": PathMNISTDataset,
        "dermamnist": DermaMNISTDataset,
        "retinamnist": RetinaMNISTDataset,
        "bloodmnist": BloodMNISTDataset,
        "breastmnist": BreastMNISTDataset,
    }
    dataset_flag = selected_dataset
    DatasetClass = dataset_classes[dataset_flag]

    try:
        train_dataset = DatasetClass(splitSet='train')
        val_dataset = DatasetClass(splitSet='val')
        test_dataset = DatasetClass(splitSet='test')

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        sys.exit(1)

    vnca = VNCA(h, w, n_channels, z_size, encoder, update_net, train_loader.dataset, val_loader.dataset, 
                test_loader.dataset, state_to_dist, batch_size, dmg_size, p_update, min_steps, max_steps)

    results_dir = os.path.join(grandparent_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    try:
        vnca.eval_batch()
    except Exception as e:
        print(f"Error during initial evaluation: {e}")

    n_updates = n_updates_s
    eval_interval = eval_interval_s
    try:
        train(vnca, n_updates, eval_interval)
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

    save_path = os.path.join(results_dir, f'vnca_model_{selected_dataset}_{n_updates}_{eval_interval}.pth')

    try:
        torch.save(vnca.state_dict(), save_path)
        print(f"Model weights saved to {save_path}")
    except Exception as e:
        print(f"Error saving model weights: {e}")
        sys.exit(1)

    try:
        vnca.test(num_test)
        print("Inference completed.")
    except Exception as e:
        print(f"Error during testing: {e}")
