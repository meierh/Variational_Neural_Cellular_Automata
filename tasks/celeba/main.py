import os
# modify
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

import torch
from torch import nn
from torch.nn import DataParallel
from torchvision import transforms, datasets

from modules.dml import DiscretizedMixtureLogitsDistribution
from modules.residual import Residual
from modules.vnca import VNCA
from train import train

dataset_name = "celeba"
n_updates_s = 50_000
eval_interval_s = 1000

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
    n_channels = 3


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

    # encoder = DataParallel(encoder)
    # update_net = DataParallel(update_net)

    data_dir = os.environ.get('DATA_DIR') or "data"
    tp = transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor()])
    train_data, val_data, test_data = [datasets.CelebA(data_dir, split=split, download=True, transform=tp) for split in ["train", "valid", "test"]]

    vnca = VNCA(h, w, n_channels, z_size, encoder, update_net, train_data, val_data, test_data, state_to_dist, batch_size, dmg_size, p_update, min_steps, max_steps)
    
    results_dir = os.path.join(grandparent_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    checkpoint_path = os.path.join(results_dir, f"checkpoint_{dataset_name}.pth")
    latest_path = os.path.join(results_dir, f"latest_{dataset_name}.pth")
    best_path = os.path.join(results_dir, f"best_{dataset_name}.pth")

    # load the latest model weights
    max_update = -1
    load_path = None

    # compare the update number of the three files
    if os.path.exists(latest_path):
        latest_update = vnca.load(latest_path)
        if latest_update > max_update:
            max_update = latest_update
            load_path = latest_path

    if os.path.exists(checkpoint_path):
        checkpoint_update = vnca.load(checkpoint_path)
        if checkpoint_update > max_update:
            max_update = checkpoint_update
            load_path = checkpoint_path

    if os.path.exists(best_path):
        best_update = vnca.load(best_path)
        if best_update > max_update:
            max_update = best_update
            load_path = best_path

    # only load the latest model weights
    if max_update == -1:
        print("\n*******************************\nNo checkpoint found, starting from scratch.\n*******************************\n")
        load_path = checkpoint_path  # default path for saving the model weights
    else:
        print(
        f"\n*******************************\n"
        f"Loading checkpoint from {os.path.relpath(load_path)} with {max_update} updates. "
        f"\nRemaining updates: {n_updates_s - max_update}.\n"
        f"*******************************\n"
    )
        vnca.load(load_path)

    try:
        vnca.eval_batch()
    except Exception as e:
        print(f"\n*******************************\nError during initial evaluation: {e}\n*******************************\n")

    n_updates = n_updates_s
    eval_interval = eval_interval_s
    try:
        train(vnca, dataset_name, n_updates, eval_interval, checkpoint_path=load_path, save_dir=results_dir)
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)
    
    save_path = os.path.join(results_dir, f'vnca_model_{dataset_name}_{n_updates}_{eval_interval}.pth')

    try:
        torch.save(vnca.state_dict(), save_path)
        print(f"Model weights saved to {os.path.relpath(save_path)}")
    except Exception as e:
        print(f"Error saving model weights: {e}")
        sys.exit(1)

    try:
        vnca.test(40)
        print("Inference completed.")
    except Exception as e:
        print(f"Error during testing: {e}")
