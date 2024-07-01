import torch
from torchvision.utils import save_image, make_grid
from torch import nn

from modules.residual import Residual
from modules.vnca import VNCA
from modules.dml import DiscretizedMixtureLogitsDistribution

# Configuration and hyperparameters
z_size = 256
nca_hid = 128
batch_size = 32
dmg_size = 16
filter_size = 5
pad = filter_size // 2
encoder_hid = 32
h = w = 32
n_channels = 3  # Obtained from main.py as pic_channels

# Define the state_to_dist function
def state_to_dist(state):
    n_mixtures = 1
    return DiscretizedMixtureLogitsDistribution(n_mixtures, state[:, :n_mixtures * 10, :, :])

# Define encoder and update_net
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

# Initialize the VNCA model
model = VNCA(
    h=h,
    w=w,
    n_channels=n_channels,
    z_size=z_size,
    encoder=encoder,
    update_net=update_net,
    train_data=None,  # No training data needed for image generation
    val_data=None,  # No validation data needed for image generation
    test_data=None,  # No test data needed for image generation
    states_to_dist=state_to_dist,
    batch_size=batch_size,
    dmg_size=dmg_size,
    p_update=1.0,
    min_steps=64,
    max_steps=128
)

input_name = '../weights/path_50k.pth'
output_name = '../images/'+input_name.split('.')[-2].split('/')[-1]+'_image.png'

# Load the trained model weights
state_dict = torch.load(input_name, map_location=torch.device('cpu'))

# Get the model state dictionary
# model_state_dict = state_dict['model_state_dict'] # For derma and retina mnist dataset
model_state_dict = state_dict # For blood and path mnist dataset

# Directly load model_state_dict
model.load_state_dict(model_state_dict)
model.eval()

# Generate 64 images with different initial noise
num_images = 64
initial_noise = torch.randn(num_images, z_size, 1, 1)  # 64 different noise with latent variable size

# Generate images using the model
with torch.no_grad():
    seeds = initial_noise.expand(-1, -1, h, w)
    states = model.decode(seeds)
    generated_images, _ = model.to_rgb(states[-1])

# Create an 8x8 grid of images
grid_image = make_grid(generated_images, nrow=8)

# Save the generated grid image
save_image(grid_image, output_name)
