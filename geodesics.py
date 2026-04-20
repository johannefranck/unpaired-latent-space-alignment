import random
import torch
import torch.nn as nn


# ============================================================
# Curve families with fixed endpoints
# ============================================================

class QuadraticCurve(nn.Module):
    """
    Quadratic-like curve with fixed endpoints:

        c(t) = z0 + (z1 - z0) t
               + a * t(1-t)
               + b * t^2(1-t)

    This ensures c(0)=z0 and c(1)=z1.
    """
    def __init__(self, z0, z1):
        super().__init__()
        self.register_buffer("z0", z0.detach().clone())
        self.register_buffer("z1", z1.detach().clone())

        d = z0.shape[0]
        self.a = nn.Parameter(torch.zeros(d, device=z0.device))
        self.b = nn.Parameter(torch.zeros(d, device=z0.device))

    def forward(self, t):
        z0 = self.z0.unsqueeze(0)
        z1 = self.z1.unsqueeze(0)
        a = self.a.unsqueeze(0)
        b = self.b.unsqueeze(0)
        t = t.unsqueeze(-1)

        line = z0 + (z1 - z0) * t
        bend1 = a * t * (1.0 - t)
        bend2 = b * (t ** 2) * (1.0 - t)

        return line + bend1 + bend2


class CubicCurve(nn.Module):
    """
    Cubic-like curve with fixed endpoints:

        c(t) = z0 + (z1 - z0) t
               + a * t(1-t)
               + b * t^2(1-t)
               + c * t(1-t)^2

    Also ensures c(0)=z0 and c(1)=z1.
    """
    def __init__(self, z0, z1):
        super().__init__()
        self.register_buffer("z0", z0.detach().clone())
        self.register_buffer("z1", z1.detach().clone())

        d = z0.shape[0]
        self.a = nn.Parameter(torch.zeros(d, device=z0.device))
        self.b = nn.Parameter(torch.zeros(d, device=z0.device))
        self.c = nn.Parameter(torch.zeros(d, device=z0.device))

    def forward(self, t):
        z0 = self.z0.unsqueeze(0)
        z1 = self.z1.unsqueeze(0)
        a = self.a.unsqueeze(0)
        b = self.b.unsqueeze(0)
        c = self.c.unsqueeze(0)
        t = t.unsqueeze(-1)

        line = z0 + (z1 - z0) * t
        bend1 = a * t * (1.0 - t)
        bend2 = b * (t ** 2) * (1.0 - t)
        bend3 = c * t * ((1.0 - t) ** 2)

        return line + bend1 + bend2 + bend3


# ============================================================
# Decoder helpers
# ============================================================

def decode_to_observation(model, z):
    """
    Decode latent points to observation space.

    Supports:
    - decoder returning a tensor
    - decoder returning a distribution-like object with tensor .mean

    For Bernoulli VAE logits, we apply sigmoid.
    """
    out = model.decode(z)

    if torch.is_tensor(out):
        x = out
        x_min = x.detach().min().item()
        x_max = x.detach().max().item()
        if x_min < 0.0 or x_max > 1.0:
            x = torch.sigmoid(x)
        return x

    if hasattr(out, "mean") and torch.is_tensor(out.mean):
        return out.mean

    raise TypeError(f"Unsupported decoder output type: {type(out)}")


def flatten_observations(x):
    if not torch.is_tensor(x):
        raise TypeError(f"Expected tensor, got {type(x)}")
    return x.view(x.shape[0], -1)


# ============================================================
# Discrete energy and length
# ============================================================

def compute_energy(curve, times, model):
    Zs = curve(times)
    Xs = decode_to_observation(model, Zs)
    Xs_flat = flatten_observations(Xs)

    diffs = Xs_flat[1:] - Xs_flat[:-1]
    dt = times[1] - times[0]

    dist_sq = diffs.pow(2).sum(dim=1)
    E = (dist_sq / dt).sum()
    return E


@torch.no_grad()
def compute_length(curve, times, model):
    Zs = curve(times)
    Xs = decode_to_observation(model, Zs)
    Xs_flat = flatten_observations(Xs)

    diffs = Xs_flat[1:] - Xs_flat[:-1]
    seg_lengths = torch.sqrt(diffs.pow(2).sum(dim=1) + 1e-12)
    L = seg_lengths.sum()
    return L


def compute_energy_ensemble(curve, times, decoders, M=10):
    Zs = curve(times)
    S = Zs.shape[0] - 1
    dt = times[1] - times[0]

    total = torch.zeros(1, device=Zs.device)

    for _ in range(M):
        segment_energy = torch.zeros(1, device=Zs.device)

        for s in range(S):
            dec_l = random.choice(decoders)
            dec_k = random.choice(decoders)

            out_l = dec_l(Zs[s:s+1])
            out_k = dec_k(Zs[s+1:s+2])

            if torch.is_tensor(out_l):
                X_l = out_l
                if X_l.detach().min().item() < 0.0 or X_l.detach().max().item() > 1.0:
                    X_l = torch.sigmoid(X_l)
            elif hasattr(out_l, "mean") and torch.is_tensor(out_l.mean):
                X_l = out_l.mean
            else:
                raise TypeError(f"Unsupported decoder output type: {type(out_l)}")

            if torch.is_tensor(out_k):
                X_k = out_k
                if X_k.detach().min().item() < 0.0 or X_k.detach().max().item() > 1.0:
                    X_k = torch.sigmoid(X_k)
            elif hasattr(out_k, "mean") and torch.is_tensor(out_k.mean):
                X_k = out_k.mean
            else:
                raise TypeError(f"Unsupported decoder output type: {type(out_k)}")

            X_l_flat = X_l.view(X_l.size(0), -1)
            X_k_flat = X_k.view(X_k.size(0), -1)

            diff = X_l_flat - X_k_flat
            segment_energy += diff.pow(2).sum() / dt

        total += segment_energy

    return total / M


# ============================================================
# Curve optimization
# ============================================================

def build_curve(z0, z1, curve_type="quadratic"):
    if curve_type == "quadratic":
        return QuadraticCurve(z0, z1)
    elif curve_type == "cubic":
        return CubicCurve(z0, z1)
    else:
        raise ValueError(f"Unknown curve_type '{curve_type}'. Use 'quadratic' or 'cubic'.")


def optimize_curve(
    model,
    z0,
    z1,
    num_steps=1000,
    S=20,
    lr=1e-1,
    ensemble=False,
    device="cpu",
    curve_type="quadratic",
    print_every=25,
):
    """
    Optimize a latent curve between z0 and z1 by minimizing discrete energy.

    Returns:
        curve      : optimized curve module
        times      : discretization times
        best_E     : best energy found
        best_L     : length of best curve
    """
    z0 = z0.to(device)
    z1 = z1.to(device)

    curve = build_curve(z0, z1, curve_type=curve_type).to(device)
    times = torch.linspace(0.0, 1.0, S + 1, device=device)

    optimizer = torch.optim.Adam(curve.parameters(), lr=lr)

    best_energy = None
    best_length = None
    best_state = None

    for step in range(num_steps):
        optimizer.zero_grad()

        if ensemble:
            E = compute_energy_ensemble(curve, times, model.decoder)
        else:
            E = compute_energy(curve, times, model)

        E.backward()
        optimizer.step()

        E_val = E.item()
        if best_energy is None or E_val < best_energy:
            best_energy = E_val
            best_state = {
                k: v.detach().clone()
                for k, v in curve.state_dict().items()
            }
            best_length = compute_length(curve, times, model).item()

        if print_every > 0 and step % print_every == 0:
            print(f"step {step:04d} | energy = {E_val:.6f}")

    curve.load_state_dict(best_state)

    if print_every > 0:
        print(f"best energy: {best_energy:.6f}")
        print(f"best length: {best_length:.6f}")

    return curve, times, best_energy, best_length


def geodesic_distance(
    model,
    z0,
    z1,
    cfg,
):
    """
    Approximate geodesic distance by:
    1) optimizing curve energy
    2) returning the length of the best optimized curve
    """
    if torch.allclose(z0, z1):
        return torch.tensor(0.0, device=z0.device)

    curve, times, _, best_length = optimize_curve(
        model=model,
        z0=z0,
        z1=z1,
        num_steps=cfg.get("steps", 1000),
        S=cfg.get("num_segments", 20),
        lr=cfg.get("lr", 1e-1),
        ensemble=cfg.get("ensemble", False),
        device=z0.device,
        curve_type=cfg.get("curve_type", "quadratic"),
        print_every=cfg.get("print_every", 0),
    )

    return torch.tensor(best_length, dtype=torch.float32, device=z0.device)


# ============================================================
# Pairwise matrices
# ============================================================

def compute_geodesics(Z, model, cfg):
    """
    Pairwise geodesic matrix using learned decoder geometry.

    Z: shape (n, latent_dim)
    returns: tensor of shape (n, n) on CPU
    """
    model.eval()
    device = next(model.parameters()).device
    Z = torch.as_tensor(Z, dtype=torch.float32, device=device)

    n = Z.shape[0]
    C = torch.zeros(n, n, dtype=torch.float32, device=device)

    for i in range(n):
        for j in range(i + 1, n):
            d = geodesic_distance(model, Z[i], Z[j], cfg)
            C[i, j] = d
            C[j, i] = d

    return C.cpu()


def compute_geodesics_S2(Z):
    """
    Exact geodesics on S^2 via arccos(inner product).
    """
    Z = torch.as_tensor(Z, dtype=torch.float32)
    Z = Z / (Z.norm(dim=1, keepdim=True) + 1e-12)
    G = torch.clamp(Z @ Z.T, -1.0, 1.0)
    return torch.arccos(G)