import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random

class QuadraticCurve(nn.Module):
    """
    A 2-parameter curve in latent space:

       c(t) = z0 + (z1 - z0)*t
              + a * [t*(1-t)]
              + b * [t^2*(1-t)]

    Ensures c(0)=z0, c(1)=z1, but let's you optimize
    a and b for more "bending" than a single parameter.
    """
    def __init__(self, z0, z1):
        """
        z0, z1: Tensors of shape (latent_dim,) => fixed endpoints
        """
        super().__init__()
        # Store endpoints (no grad)
        self.register_buffer("z0", z0.detach().clone())
        self.register_buffer("z1", z1.detach().clone())

        # 2 param vectors => each dimension has 2 degrees of freedom
        M = z0.shape[0]
        self.a = nn.Parameter(torch.zeros(M))
        self.b = nn.Parameter(torch.zeros(M))

    def forward(self, t):
        """
        Evaluate c(t) for a scalar or vector t in [0,1].
        If t is shape (N,) => return shape (N, M).
        """
        z0 = self.z0.unsqueeze(0)  # (1, M)
        z1 = self.z1.unsqueeze(0)  # (1, M)
        a = self.a.unsqueeze(0)    # (1, M)
        b = self.b.unsqueeze(0)
        t  = t.unsqueeze(-1)       # (N, 1)

        line = z0 + (z1 - z0)*t      # shape (N, M)
        bend1 = a * t * (1 - t)      # shape (N, M)
        bend2 = b * (t**2) * (1 - t) # shape (N, M)
        
        return line + bend1 + bend2

def compute_energy(curve, times, decoder):
    """
    Compute the energy E[c] = sum_{s=1..S} || f(c_s) - f(c_{s-1}) ||^2
    where c_s = curve(times[s]) in latent space, and f is the decoder.
    
    times: a 1D tensor of S+1 points in [0,1], e.g. torch.linspace(0,1,S+1).
    decoder: VAE's decoder, mapping latent points to data space.
    """
    # Evaluate the curve at the discrete times
    Zs = curve(times)  # shape (S+1, latent_dim)

    # Map each latent point to data space via decoder
    # We'll use the mean for a deterministic path
    Xs = decoder(Zs).mean  # shape (S+1, 1, 28, 28) for MNIST

    # Flatten to compute Euclidean distances in (pixel) data space:
    Xs_flat = Xs.view(Xs.size(0), -1)  # (S+1, 784) for MNIST 28x28

    diffs = Xs_flat[1:] - Xs_flat[:-1]   # shape (S, 784)
    dist_sq = diffs.pow(2).sum(dim=1)    # sum over pixels
    E = dist_sq.sum()                    # final scalar

    return E

def compute_energy_ensemble(curve, times, decoders, M=10):
    # Evaluate the curve at the discrete times: shape (S+1, latent_dim)
    Zs = curve(times)
    S = Zs.shape[0] - 1

    # Initialize the total energy as a tensor on the same device as Zs.
    total = torch.zeros(1, device=Zs.device)

    # Monte Carlo: for each iteration, sample a decoder pair for each segment independently.
    for _ in range(M):
        segment_energy = torch.zeros(1, device=Zs.device)
        # For each segment, independently sample a decoder pair.
        for s in range(S):
            dec_l = random.choice(decoders)
            dec_k = random.choice(decoders)
            
            # Evaluate decoder outputs on the endpoints of the segment.
            X_l = dec_l(Zs[s:s+1]).mean   # shape (1, 1, 28, 28) for MNIST
            X_k = dec_k(Zs[s+1:s+2]).mean  # shape (1, 1, 28, 28)
            
            # Flatten to vectors (shape (1, num_pixels)).
            X_l_flat = X_l.view(X_l.size(0), -1)
            X_k_flat = X_k.view(X_k.size(0), -1)
            
            # Compute squared Euclidean distance for the segment.
            diff = X_l_flat - X_k_flat
            segment_energy += diff.pow(2).sum()
        
        total += segment_energy

    # Average the energy over the M Monte Carlo samples.
    energy_estimate = total / M
    
    return energy_estimate

def optimize_curve(model, z0, z1, num_steps=1000, S=20, lr = 1e-1, ensemble = False, device="cpu"):
    
    # 1. Build our QuadraticCurve
    curve = QuadraticCurve(z0.to(device), z1.to(device)).to(device)

    # 2. Choose discrete times in [0,1]
    times = torch.linspace(0, 1, S+1, device=device)

    # 3. Optimizer for the curve
    optimizer = optim.Adam(curve.parameters(), lr=lr)

    for step in range(num_steps):
        
        optimizer.zero_grad()
        
        if ensemble == True: 
            E = compute_energy_ensemble(curve, times, model.decoder)
        else:
            E = compute_energy(curve, times, model.decoder)
        
        E.backward()
        optimizer.step()
  
        if step % 25 == 0:
            print(f"Step {step}: E = {E.item():.4f}")

    print(f"Final energy: {E.item():.4f}")
    return curve

def Compute_CoV(distances): 
    """
    Coefficient of Variation (CoV) = std / mean
    """
    mean = np.mean(distances)
    std = np.std(distances)
    cov = std / mean
    return cov

