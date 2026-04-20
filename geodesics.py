import torch
import torch.nn as nn


class QuadraticCurve(nn.Module):
    """
    Quadratic latent curve with fixed endpoints:
        c(0) = z_start, c(1) = z_end
    """
    def __init__(self, z_start, z_end):
        super().__init__()
        self.register_buffer("z_start", z_start.detach().clone())
        self.register_buffer("z_end", z_end.detach().clone())

        d = z_start.shape[0]
        self.a = nn.Parameter(torch.zeros(d))
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, t):
        z0 = self.z_start.unsqueeze(0)
        z1 = self.z_end.unsqueeze(0)
        a = self.a.unsqueeze(0)
        b = self.b.unsqueeze(0)
        t = t.unsqueeze(-1)

        return z0 + (z1 - z0) * t + a * t * (1 - t) + b * t**2 * (1 - t)


def decode(model, z):
    """
    Decode latent → observation space (Bernoulli probs).
    """
    return torch.sigmoid(model.decode(z))


def curve_energy(curve, t_grid, model):
    """
    Discrete path energy in observation space.
    """
    z = curve(t_grid)
    x = decode(model, z).view(z.shape[0], -1)

    diffs = x[1:] - x[:-1]
    return (diffs**2).sum()


def geodesic_distance(model, z0, z1, cfg):
    """
    Approximate geodesic distance via energy minimization.
    """
    curve = QuadraticCurve(z0, z1)
    opt = torch.optim.Adam(curve.parameters(), lr=cfg["lr"])

    t_grid = torch.linspace(0, 1, cfg["num_segments"] + 1, device=z0.device)

    for _ in range(cfg["steps"]):
        opt.zero_grad()
        E = curve_energy(curve, t_grid, model)
        E.backward()
        opt.step()

    return torch.sqrt(torch.clamp(E, min=1e-12))


def compute_geodesics(Z, model, cfg):
    """
    Pairwise geodesic matrix using learned decoder geometry.
    """
    Z = torch.as_tensor(Z, dtype=torch.float32)
    n = Z.shape[0]

    C = torch.zeros(n, n, dtype=torch.float32)

    for i in range(n):
        for j in range(i + 1, n):
            d = geodesic_distance(model, Z[i], Z[j], cfg)
            C[i, j] = d
            C[j, i] = d

    return C


def compute_geodesics_S2(Z):
    """
    Exact geodesics on S^2 via arccos(inner product).
    """
    Z = torch.as_tensor(Z, dtype=torch.float32)
    Z = Z / (Z.norm(dim=1, keepdim=True) + 1e-12)

    return torch.arccos(torch.clamp(Z @ Z.T, -1.0, 1.0))