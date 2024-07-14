import os
import sys
import datetime
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
from modules.loss import elbo, iwae, mse_loss, bce_loss, kl_divergence_loss, hinge_loss, l1_loss, smooth_l1_loss # possible loss functions
from modules.vnca import VNCA
from train import train
import torch

selected_dataset = "dermamnist" # choose from "pathmnist", "dermamnist", "retinamnist", "bloodmnist", "breastmnist
pic_channels = 3
n_updates_s = 50_000
eval_interval_s = 1000
num_test = 40 # use 

# Experiment settings and results
# Choose the loss function from the possible loss functions above
loss_fn_t = elbo # modified for different losses, originally elbo
loss_fn_e = iwae # modified for different losses, originally iwae
loss_sel = "elbo_train_iwae_eval" # put by the end of the pth file name
# test resuts train use mse_loss and eval use iwae
# when use mse_loss will after 6 epochs already have non-finite gradient problem
# when use bce_loss will after 8 epochs already have non-finite gradient problem
# when use kl_divergence_loss will after 10 epochs already have non-finite gradient problem

# Choose the filter size
filter_size_t = 3  # modified for different filters sizes, originally 5
# test resuts for training with different filter sizes

# chosse learning rate
lr_s = 1e-4  # modified for different learning rates, originally 1e-4
# try 1e-5, 1e-6, 1e-7, they cant avoid gradient going to non-finite problem, just delayed it, the lower the lr the longer it takes to reach the non-finite gradient problem

if __name__ == "__main__":
    z_size = 256
    nca_hid = 128
    n_mixtures = 1
    batch_size = 32
    dmg_size = 16
    p_update = 1.0
    min_steps, max_steps = 64, 128

    filter_size = filter_size_t
    pad = filter_size // 2
    print(f"Pad size: {pad}") 
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

    # Iterate through the layers and print the parameters
    for name, layer in encoder.named_children():
        print(f"Layer: {name}")
        for param in layer.parameters():
            print(param.shape)

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
                test_loader.dataset, state_to_dist, batch_size, dmg_size, p_update, min_steps, max_steps, loss_fn_t, loss_fn_e, lr_s)  # modified for different losses

    results_dir = os.path.join(grandparent_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    checkpoint_path = os.path.join(results_dir, f"checkpoint_{selected_dataset}.pth")
    latest_path = os.path.join(results_dir, f"latest_{selected_dataset}.pth")
    best_path = os.path.join(results_dir, f"best_{selected_dataset}.pth")

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
        train(vnca, selected_dataset, n_updates, eval_interval, checkpoint_path=load_path, save_dir=results_dir)
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

    save_path = os.path.join(results_dir, f'vnca_model_{selected_dataset}_{n_updates}_{eval_interval}_{loss_sel}_filter_size_{filter_size_t}_lr_{lr_s}.pth')

    try:
        torch.save(vnca.state_dict(), save_path)
        print(f"Model weights saved to {os.path.relpath(save_path)}")
    except Exception as e:
        print(f"Error saving model weights: {e}")
        sys.exit(1)

    try:
        vnca.test(num_test)
        print("Inference completed.")
    except Exception as e:
        print(f"Error during testing: {e}")
