import torch
from torchvision.utils import save_image, make_grid
from torch import nn
import cv2
import numpy as np

from modules.residual import Residual
from modules.vnca import VNCA
from modules.dml import DiscretizedMixtureLogitsDistribution

# Configuration and hyperparameters
z_size = 256
nca_hid = 128
batch_size = 1  # Set batch size to 1 to generate a single image
dmg_size = 16
filter_size = 5
pad = filter_size // 2
encoder_hid = 32
n_channels = 3  # For color images

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
    nn.Linear(encoder_hid * (2 ** 4) * 2 * 2, 2 * z_size),
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
    h=32,  # Initial height of 32
    w=32,  # Initial width of 32
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

input_name = '../weights/derma_50k.pth'
output_name_base = '../images/' + input_name.split('.')[-2].split('/')[-1] + '_image'

# Load the trained model weights
state_dict = torch.load(input_name, map_location=torch.device('cpu'))

# Get the model state dictionary
model_state_dict = state_dict['model_state_dict']  # For derma and retina mnist dataset
# model_state_dict = state_dict # For blood and path mnist dataset

# Directly load model_state_dict
model.load_state_dict(model_state_dict)
model.eval()

# Create a 32x32 initial noise
initial_size = 32
initial_state = torch.randn(1, z_size, initial_size, initial_size)

# Print initial state for debugging
print("Initial state shape:", initial_state.shape)

# Number of NCA steps for generating the image
steps_per_iteration = 20  # Adjust based on how evolved you want the image to be

# Function for denoising the image using averaging
def denoise_image(image):
    # Convert the image to a numpy array
    image_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)

    # Apply averaging filter
    denoised_image_np = cv2.blur(image_np, (5, 5))

    # Convert back to a tensor
    denoised_image = torch.tensor(denoised_image_np).permute(2, 0, 1)

    return denoised_image

# Generate and save 64 images
all_images = []
with torch.no_grad():
    for i in range(16):
        state = initial_state.clone()
        for step in range(steps_per_iteration):
            state = state.contiguous()  # Ensure the state is contiguous
            states = model.decode(state)  # Ensure the states are a list of 4D tensors
            state = states[-1]  # Use the last state for the next iteration

        # Generate and denoise the image
        generated_images, _ = model.to_rgb(state)
        denoised_image = denoise_image(generated_images)
        all_images.append(denoised_image)

# Stack all images into a single 8x8 grid and save it
grid = make_grid(all_images, nrow=4, padding=2, normalize=True)
output_name = f'{output_name_base}_grid.png'
save_image(grid, output_name)
print(f'Saved final image grid as {output_name}')
