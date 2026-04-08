import numpy as np
import scipy
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import vonmises_fisher
from scipy.spatial.distance import pdist, squareform

from colors import color_segment

blue_pink = color_segment()


# -------------------
# General functions overview:
# -------------------

# get_rotation_matrix(params):

# construct_sphere_mesh(n_resolution):

# deform_sphere(points_on_sphere, noise_params):

# compute_geodesics(points, is_sphere=True):

# sample_vmf_on_sphere(mu, kappa, n_samples):

# gromov_wasserstein_loss(D1, D2, P):

# rotation_alignment_loss(R, points1, points2, P):

# solve_sinkhorn_coupling(cost_matrix, epsilon, a, b, pairs=None):

# optimize_rotation(initial_params, points1, points2, coupling_P, pairs=None):

# plot_object_points(mesh, points, point_colors, ax):
   


def get_rotation_matrix(theta):
    """
    Generates a 3x3 rotation matrix using the matrix exponential of a skew-symmetric matrix.
    Inputs: theta (array of 3 angles)
    Outputs: R (3x3 rotation matrix)
    """
    tx, ty, tz = theta
    # Skew-symmetric matrix representation of the rotation vector
    A = np.array([[0, -tz, ty],
                  [tz, 0, -tx],
                  [-ty, tx, 0]])
    return scipy.linalg.expm(A)

def construct_sphere_mesh(n_resolution):
    """
    Objective: Create vertices and faces for a standard unit sphere mesh (S2).
    Logic: Uses spherical coordinates discretized by n_resolution.
    Inputs: n_resolution (int)
    Outputs: vertices (N x 3), faces (M x 3)
    """
    # Create a grid of phi (0 to pi) and theta (0 to 2*pi)
    phi = np.linspace(0, np.pi, n_resolution)
    theta = np.linspace(0, 2 * np.pi, n_resolution)
    phi_grid, theta_grid = np.meshgrid(phi, theta)

    # Convert spherical coordinates to Cartesian coordinates for the unit sphere
    x = np.sin(phi_grid) * np.cos(theta_grid)
    y = np.sin(phi_grid) * np.sin(theta_grid)
    z = np.cos(phi_grid)

    # Flatten grids to create N x 3 vertex array
    vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    # Generate face indices for the grid structure
    faces = []
    for i in range(n_resolution - 1):
        for j in range(n_resolution - 1):
            # Define indices for the corners of each quad on the grid
            p1 = i * n_resolution + j
            p2 = p1 + 1
            p3 = (i + 1) * n_resolution + j
            p4 = p3 + 1
            # Split quad into two triangles
            faces.append([p1, p2, p3])
            faces.append([p2, p4, p3])
            
    return vertices, np.array(faces)

def create_deformation_params(seed, n_terms=4):
    rng = np.random.RandomState(seed)

    params = []
    for _ in range(n_terms):
        a = rng.uniform(0.2, 0.6)
        w = rng.uniform(1.0, 4.0, size=3)
        phase = rng.uniform(-np.pi, np.pi)

        params.append((a, w, phase))

    return params

def smooth_field(points, params):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    base = np.zeros(len(points))

    for a, w, phase in params:
        base += a * np.sin(w[0]*x + w[1]*y + w[2]*z + phase)

    return base

def deform_sphere(points_on_sphere, alpha, params):
    """
    Objective: Implement the push-forward map f: S2 -> Manifold
    Logic: Implements the push-forward operator by moving point positions
           Perturbs the radial component so that |z| != 1 using non-symmetric noise.
    Inputs: points_on_sphere (N x 3), noise_params (dict with 'amplitude' and 'freq')
    Outputs: deformed_points (N x 3)
    """
    base = smooth_field(points_on_sphere, params)

    radial_scale = 1.0 + alpha * 0.25 * base

    return points_on_sphere * radial_scale[:, None]


def deform_mesh(mesh, alpha, params):
    vertices, faces = mesh
    return deform_sphere(vertices, alpha, params), faces


# --- compute geodesics ---

def compute_geodesics(points, is_sphere=True):
    """
    Computes the pairwise intra-distance matrix D.
    Inputs: points (N x 3), is_sphere (bool toggle)
    Outputs: D (N x N distance matrix)

    Used in the initial GW4 experiments only.
    """
    if is_sphere:
        # Vectorized spherical distance: arccos(Xi * Xj)
        # Using np.clip to prevent numerical errors outside [-1, 1]
        inner_prod = np.inner(points, points)
        return np.arccos(np.clip(inner_prod, -1.0, 1.0))
    else:
        # Euclidean approximation for deformed objects
        # Using scipy's efficient pdist/squareform for ||xi - xj||
        return squareform(pdist(points, 'euclidean'))
    


def compute_pairwise_distances(points):
    return squareform(pdist(points, metric="euclidean"))


def solve_gw_coupling(D1, D2, epsilon=0.1, threshold=1e-6, max_iter=200):
    D1 = np.array(D1, dtype=np.float64, copy=True)
    D2 = np.array(D2, dtype=np.float64, copy=True)

    D1 = D1 / (D1.max() + 1e-12)
    D2 = D2 / (D2.max() + 1e-12)

    n = D1.shape[0]
    m = D2.shape[0]

    a = np.ones(n) / n
    b = np.ones(m) / m

    P = a[:, None] @ b[None, :]
    gw_old = np.inf

    for _ in range(max_iter):
        cost_gradient = -D1 @ P @ D2
        P = solve_sinkhorn_coupling(cost_gradient, epsilon, a, b)

        gw_new = gromov_wasserstein_loss(D1, D2, P)

        if abs(gw_new - gw_old) < threshold:
            break

        gw_old = gw_new

    return P, gw_new


def fit_orthogonal_map(X_src, X_tgt):
    """
    Solves min_R ||X_tgt R - X_src||_F^2  subject to R^T R = I
    """
    Xs = X_src - X_src.mean(axis=0, keepdims=True)
    Xt = X_tgt - X_tgt.mean(axis=0, keepdims=True)

    M = Xt.T @ Xs
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    t = X_src.mean(axis=0) - X_tgt.mean(axis=0) @ R
    return R, t


def fit_orthogonal_map_from_coupling(X_src, X_tgt, P):
    """
    Solves min_{R,t} sum_ij P_ij ||x_i - (y_j R + t)||^2
    with R orthogonal.
    """
    a = P.sum(axis=1)
    b = P.sum(axis=0)

    mu_src = a @ X_src
    mu_tgt = b @ X_tgt

    Xc = X_src - mu_src
    Yc = X_tgt - mu_tgt

    M = Yc.T @ P.T @ Xc
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    t = mu_src - mu_tgt @ R
    return R, t


def apply_affine_map(X, R, t):
    return X @ R + t

def fit_affine_map_from_coupling(X_src, X_tgt, P, reg=1e-8):
    """
    Objective: Fit an affine map x -> xW + t from target to source using a soft coupling.
    Logic: Solves a weighted least-squares problem where the coupling P defines
           how strongly each target point should match each source point.
    Inputs: X_src (N x d), X_tgt (M x d), P (N x M), reg (small ridge parameter)
    Outputs: W (d x d), t (d,)
    """
    # Extract target-side marginal
    b = P.sum(axis=0)

    # Compute the barycentric target image in source space
    # For each target point y_j, this gives the weighted average source point
    Y_bar = (P.T @ X_src) / (b[:, None] + 1e-16)

    # Keep only points with non-negligible transported mass
    mask = b > 1e-12
    X = X_tgt[mask]
    Y = Y_bar[mask]
    w = b[mask]

    # Build augmented design matrix for affine fitting
    X_aug = np.concatenate([X, np.ones((len(X), 1))], axis=1)

    # Weighted least-squares normal equations
    W_diag = np.diag(w)
    A = X_aug.T @ W_diag @ X_aug
    B = X_aug.T @ W_diag @ Y

    # Small ridge regularization for stability
    A = A + reg * np.eye(A.shape[0])

    Theta = np.linalg.solve(A, B)

    # Split into linear part and translation
    W = Theta[:-1, :]
    t = Theta[-1, :]

    return W, t

def learn_labelwise_gw_maps(zA, zB, y, labels=None, epsilon=0.02, threshold=1e-6):
    """
    Objective: Learn one class-conditional GW map per label.
    Logic: For each label, first solve a GW coupling between the two latent
           distributions, then fit an affine map from the target latent space
           into the source latent space using the learned coupling.
    Inputs: zA (N x d), zB (N x d), y (N,), labels (optional), epsilon, threshold
    Outputs: maps (dict), couplings (dict), losses (dict)
    """
    if labels is None:
        labels = np.unique(y)

    maps = {}
    couplings = {}
    losses = {}

    for k in labels:
        XA = zA[y == k]
        XB = zB[y == k]

        D_A = compute_pairwise_distances(XA)
        D_B = compute_pairwise_distances(XB)

        P, gw_loss = solve_gw_coupling(
            D_A,
            D_B,
            epsilon=epsilon,
            threshold=threshold
        )

        W, t = fit_affine_map_from_coupling(XA, XB, P)

        maps[k] = {"W": W, "t": t}
        couplings[k] = P
        losses[k] = gw_loss

    return maps, couplings, losses


def apply_labelwise_maps(z, y, maps):
    """
    Objective: Apply the learned class-conditional affine maps to a latent dataset.
    Logic: Uses the label of each point to choose the corresponding learned map.
    Inputs: z (N x d), y (N,), maps (dict)
    Outputs: z_aligned (N x d)
    """
    z_aligned = np.zeros_like(z)

    for k, params in maps.items():
        mask = (y == k)
        W = params["W"]
        t = params["t"]
        z_aligned[mask] = z[mask] @ W + t

    return z_aligned

# --- distributions ---

def sample_vmf_on_sphere(mu, kappa, n_samples):
    """
    Objective: Properly sample from a Von Mises-Fisher distribution on the unit sphere.
    Logic: Ensures the mean direction is normalized and uses scipy to sample 
           directly on the S2 manifold where ||z|| = 1.
    Inputs: mu (mean direction), kappa (concentration), n_samples (int)
    Outputs: points (n_samples x 3) where ||z|| = 1
    """
    # Ensure the mean direction is a unit vector
    mu_normalized = mu / np.linalg.norm(mu)
    
    # Sample directly from the VMF distribution on the sphere
    points = vonmises_fisher.rvs(mu_normalized, kappa, size=n_samples)
    
    return points

def symmetric_samples_vmf(mu, layer_radii, points_per_layer):
    """
    Construct symmetric vMF-like samples on S^2 around mean direction mu.

    Parameters
    ----------
    mu : (3,) array
        Mean direction (will be normalized)
    layer_radii : list or array
        Radii (angular offsets) for each ring
    points_per_layer : int
        Number of points per ring

    Returns
    -------
    data : (N,3) array
        Symmetric samples on the sphere
    """
    mu = mu.astype(float)
    mu = mu / np.linalg.norm(mu)

    # Build orthonormal basis (v1, v2) ⟂ mu
    if np.allclose(mu, np.array([1., 0., 0.])):
        v1 = np.array([0., 1., 0.])
    else:
        v1 = np.cross(mu, np.array([1., 0., 0.]))
        if np.linalg.norm(v1) < 1e-8:
            v1 = np.cross(mu, np.array([0., 1., 0.]))

    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(mu, v1)

    data_points = []

    for r in layer_radii:
        angles = np.linspace(0, 2*np.pi, points_per_layer, endpoint=False)

        layer = (
            mu[None, :]
            + r * np.cos(angles)[:, None] * v1[None, :]
            + r * np.sin(angles)[:, None] * v2[None, :]
        )

        # Project to sphere
        layer /= np.linalg.norm(layer, axis=1, keepdims=True)
        data_points.append(layer)

    return np.vstack(data_points)


# --- compute loss ---

def gromov_wasserstein_loss(D1, D2, P):
    """
    Objective: Calculate the GW objective discrepancy between two intra-distance matrices.
    Logic: Uses an efficient matrix expansion of the quadratic objective:
           Loss = sum_{i,j,i',j'} |D1_ii' - D2_jj'|^2 * P_ij * P_i'j'
           Expanded: C1 + C2 - 2 * (D1 @ P @ D2).
    Inputs: D1 (N x N), D2 (M x M), P (N x M coupling matrix)
    Outputs: gw_loss (float)
    """
    # Extract marginals from the coupling matrix P
    # a: mass distribution on the source manifold
    # b: mass distribution on the target manifold
    a = np.sum(P, axis=1)
    b = np.sum(P, axis=0)

    # Compute constant terms relative to the distance matrices and marginals
    # These represent the 'self-similarities' of each space
    C1 = np.dot(np.dot(D1**2, a[:, np.newaxis]), np.ones((1, len(b))))
    C2 = np.dot(np.ones((len(a), 1)), np.dot(b[np.newaxis, :], (D2**2).T))

    # Compute the cross-term (linearized gradient term)
    # This captures how the pairwise distances align across the manifolds
    cross_term = 2 * np.dot(D1, np.dot(P, D2))

    # The full cost matrix for the current coupling P
    cost_matrix = C1 + C2 - cross_term

    # Final discrepancy is the sum of cost weighted by the transport plan
    return np.sum(cost_matrix * P)


def rotation_alignment_loss(R, points1, points2, P):
    """
    Objective: Compute the L2 alignment loss for rotated points under a given coupling.
    Logic: Transports mass from 'points1' to the rotated 'points2' according to P.
    Inputs: R (3x3), points1 (N x 3), points2 (M x 3), P (N x M coupling)
    Outputs: alignment_loss (float)
    """
    # Apply the candidate rotation matrix R to the target point cloud
    rotated_points2 = (R @ points2.T).T

    # Compute pairwise squared Euclidean distances between source and rotated target
    diff = points1[:, np.newaxis, :] - rotated_points2[np.newaxis, :, :]
    squared_euclidean_cost = np.sum(diff**2, axis=2)

    # The loss is the total work required to move source mass to aligned target mass
    return np.sum(squared_euclidean_cost * P)



# --- optimizer ---

def solve_sinkhorn_coupling(cost_matrix, epsilon, a, b, pairs=None, n_iters=200):
    """
    Objective: Solve for the optimal coupling matrix P using Sinkhorn iterations.
    Logic: Implements entropic OT in a numerically stabilized form. In paired experiments,
           modifies the cost matrix to lock anchor mass between specific indices.
    Inputs: cost_matrix (N x M), epsilon (reg parameter), a/b (marginals),
            pairs (list of tuples, optional), n_iters (int)
    Outputs: P (optimal coupling matrix)
    """
    # Work on a float copy of the cost matrix to ensure integrity
    C = np.array(cost_matrix, dtype=np.float64, copy=True)

    # If pairs are provided, bias the cost matrix
    # This acts as a hard constraint for the anchor point
    if pairs is not None:
        large_penalty = 1e2
        for i, j in pairs:
            # Penalize all other matches for these rows and columns
            C[i, :] = large_penalty
            C[:, j] = large_penalty
            # Set zero cost for the ground-truth correspondence
            C[i, j] = 0.0

    # Shift the costs so the minimum is zero
    # This keeps the exponentials in a safer numerical range
    C = C - C.min()

    # Compute the stabilized Gibbs kernel
    # Clipping prevents overflow/underflow in exp
    scaled_cost = -C / epsilon
    scaled_cost = np.clip(scaled_cost, -700, 50)
    K = np.exp(scaled_cost)

    # Avoid exact zeros in the kernel
    # This prevents divisions by zero in the Sinkhorn updates
    K = np.maximum(K, 1e-300)

    # Initialize scaling vectors
    n, m = C.shape
    u = np.ones(n) / n
    v = np.ones(m) / m

    # Perform Sinkhorn iterations to satisfy marginal constraints
    # These iterations consist of simple matrix-vector products
    for _ in range(n_iters):
        # Update v (target scaling)
        KTu = K.T @ u
        KTu = np.maximum(KTu, 1e-300)
        v = b / KTu

        # Update u (source scaling)
        Kv = K @ v
        Kv = np.maximum(Kv, 1e-300)
        u = a / Kv

    # Reconstruct the optimal coupling matrix P = diag(u) K diag(v)
    P = u[:, np.newaxis] * K * v[np.newaxis, :]

    # Normalize once at the end to reduce accumulated numerical drift
    P = P / (P.sum() + 1e-16)

    return P


def optimize_rotation(initial_params, points1, points2, coupling_P, pairs=None):
    """
    Objective: Wrapper for scipy.optimize.minimize to find the best rotation R.
    Logic: Minimizes rotation_alignment_loss. If pairs exist, it initializes theta 
           to align anchors, helping the optimizer avoid local minima in symmetric spaces.
    Inputs: initial_params (array), points1 (N x 3), points2 (M x 3), coupling_P (N x M), pairs (list, optional)
    Outputs: optimal_theta (array of 3 angles)
    """
    start_theta = np.copy(initial_params)
    
    # Symmetry Breaking: If a pair is provided, use it to improve the initial guess
    # For S2, fixing one point leaves only rotation around that axis as a degree of freedom
    if pairs is not None and len(pairs) > 0:
        # Use the first pair as a geometric anchor
        idx1, idx2 = pairs
        v_src = points1[idx1]
        v_tgt = points2[idx2]
        
        # Calculate a rotation vector that aligns the target anchor with the source
        axis = np.cross(v_tgt, v_src)
        norm = np.linalg.norm(axis)
        if norm > 1e-6:
            axis /= norm
            angle = np.arccos(np.clip(np.dot(v_src, v_tgt), -1.0, 1.0))
            start_theta = axis * angle

    # Define the objective wrapper for the optimizer
    def objective(theta):
        # get_rotation_matrix uses matrix exponential to ensure a valid SO(3) matrix
        R = get_rotation_matrix(theta)
        return rotation_alignment_loss(R, points1, points2, coupling_P)

    # Use L-BFGS-B to find the optimal rotation parameters (theta_x, theta_y, theta_z)
    res = minimize(objective, start_theta, method='L-BFGS-B')
    
    return res.x




# --- add these to utils_GW.py ---

import torch
import torch.nn as nn


def coupling_to_barycentric_targets(X_src, X_tgt, P, eps=1e-12):
    """
    Build soft targets in source space from a GW coupling.

    Inputs
    ------
    X_src : (N, d) source anchors, e.g. zA_anchor
    X_tgt : (M, d) target anchors, e.g. zB_anchor
    P     : (N, M) coupling matrix

    Returns
    -------
    X_tgt : (M, d)
    Y_bar : (M, d), barycentric targets in source space
    """
    X_src = np.asarray(X_src, dtype=np.float64)
    X_tgt = np.asarray(X_tgt, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)

    col_mass = P.sum(axis=0, keepdims=True)
    col_mass = np.maximum(col_mass, eps)

    Y_bar = (P.T @ X_src) / col_mass.T
    return X_tgt.astype(np.float32), Y_bar.astype(np.float32)


class ResidualMap(nn.Module):
    """
    T_theta(z) = z + u_theta(z)
    """
    def __init__(self, latent_dim, hidden_dim=32):
        super().__init__()
        self.displacement = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z):
        return z + self.displacement(z)

    def displacement_field(self, z):
        return self.displacement(z)


def _pairwise_distance_distortion_loss(z_in, z_out):
    """
    Penalize distortion of pairwise Euclidean distances.
    """
    Din = torch.cdist(z_in, z_in, p=2)
    Dout = torch.cdist(z_out, z_out, p=2)
    return ((Dout - Din) ** 2).mean()


def train_residual_map_from_coupling(
    X_src,
    X_tgt,
    P,
    hidden_dim=32,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=2000,
    lambda_disp=1e-2,
    lambda_geom=1e-1,
    device="cpu",
    verbose=False,
):
    """
    Train a constrained nonlinear map from target -> source using GW barycentric targets.

    Objective
    ---------
    fit  : map target anchors to GW barycentric targets in source space
    disp : keep displacement u_theta(z) small
    geom : discourage collapse by preserving pairwise distances in target space
    """
    X_tgt_np, Y_bar_np = coupling_to_barycentric_targets(X_src, X_tgt, P)

    X_tgt_t = torch.tensor(X_tgt_np, dtype=torch.float32, device=device)
    Y_bar_t = torch.tensor(Y_bar_np, dtype=torch.float32, device=device)

    model = ResidualMap(latent_dim=X_tgt_t.shape[1], hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        Y_pred = model(X_tgt_t)
        U = model.displacement_field(X_tgt_t)

        loss_fit = mse(Y_pred, Y_bar_t)
        loss_disp = (U ** 2).mean()
        loss_geom = _pairwise_distance_distortion_loss(X_tgt_t, Y_pred)

        loss = loss_fit + lambda_disp * loss_disp + lambda_geom * loss_geom
        loss.backward()
        optimizer.step()

        if verbose and ((epoch + 1) % 200 == 0 or epoch == 0):
            print(
                f"[ResidualMap epoch {epoch+1:4d}] "
                f"loss={loss.item():.6f} "
                f"fit={loss_fit.item():.6f} "
                f"disp={loss_disp.item():.6f} "
                f"geom={loss_geom.item():.6f}"
            )

    return model


def apply_residual_map(model, X, device="cpu"):
    """
    Apply learned residual map to any set of target-space points.
    """
    X_t = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        Y = model(X_t).cpu().numpy()
    return Y