import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion


model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # Move model to GPU if available


diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000    # number of steps
)
diffusion = diffusion.to(device)

training_images = torch.rand(8, 3, 128, 128).to(device) # images are normalized from 0 to 1
loss = diffusion(training_images)
loss.backward()

sampled_images = diffusion.sample(batch_size = 4)
sampled_images.shape # (4, 3, 128, 128)
