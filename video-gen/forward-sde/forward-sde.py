import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------
# 1. Define SDE drift/diffusion functions
# ----------------------------------------

def vp_sde(x, t, beta_min=0.1, beta_max=20.0):
    """Variance Preserving SDE: dx = -1/2 * beta(t) x dt + sqrt(beta(t)) dW"""
    beta_t = beta_min + t * (beta_max - beta_min)
    drift = -0.5 * beta_t[:, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion


def ve_sde(x, t, sigma_min=0.01, sigma_max=50.0):
    """Variance Exploding SDE: dx = sigma(t) * sqrt(2 log(sigma_max/sigma_min)) dW"""
    log_sigma = torch.log(torch.tensor(sigma_max)) - torch.log(torch.tensor(sigma_min))
    sigma_t = sigma_min * torch.exp(t * log_sigma)
    drift = torch.zeros_like(x)
    diffusion = sigma_t * torch.sqrt(torch.tensor(2.0) * log_sigma)
    return drift, diffusion


# ----------------------------------------
# 2. Euler-Maruyama integrator
# ----------------------------------------

def euler_maruyama_step(x, drift, diffusion, dt):
    noise = torch.randn_like(x)
    return x + drift * dt + diffusion[:, None] * torch.sqrt(dt) * noise


# ----------------------------------------
# 3. Simulate forward SDE
# ----------------------------------------

def simulate_forward_sde(sde_func, steps=1000, dt=1/1000, N=1024):
    # initial data distribution = Gaussian mixture to visualize smoothing
    x = torch.cat([
        torch.randn(N//2, 2) * 0.3 + torch.tensor([[2.0, 0.0]]),
        torch.randn(N//2, 2) * 0.3 + torch.tensor([[-2.0, 0.0]])
    ], dim=0).to(device)

    xs = [x.cpu().numpy()]
    t = torch.zeros(N).to(device)

    for step in range(steps):
        drift, diffusion = sde_func(x, t)
        x = euler_maruyama_step(x, drift, diffusion, dt)
        t = t + dt
        if step % (steps // 10) == 0:
            xs.append(x.cpu().numpy())
    
    return xs
