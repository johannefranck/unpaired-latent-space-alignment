import numpy as np
import scipy
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import vonmises_fisher
from scipy.spatial.distance import pdist, squareform

from colors import color_segment
from utils_GW import get_rotation_matrix

blue_pink = color_segment()


# --- plotting functions ---

# (KDE only for plotting)
def spherical_kde_density(points, kappa=40, res=200):
    """
    Kernel density estimate on S^2 using vMF kernels centered at data points.
    """
    theta = np.linspace(0, np.pi, res)
    phi = np.linspace(0, 2*np.pi, res)
    Theta, Phi = np.meshgrid(theta, phi)

    X = np.stack([
        np.cos(Phi) * np.sin(Theta),
        np.sin(Phi) * np.sin(Theta),
        np.cos(Theta)
    ], axis=-1)   # (res,res,3)

    pts = points / np.linalg.norm(points, axis=1, keepdims=True)

    dot = np.einsum("...i,ni->...n", X, pts)

    density = np.exp(kappa * dot).mean(axis=-1)

    density /= density.max()

    return Theta, Phi, density

def add_density_contours_s2(ax3d, points, levels=(0.4,0.6,0.8), res=200,
                           kappa=40, color="gray", alpha=0.35, linewidth=2):

    Theta, Phi, density = spherical_kde_density(points, kappa=kappa, res=res)

    fig2, ax2d = plt.subplots()
    CS = ax2d.contour(Theta, Phi, density, levels=list(levels))
    plt.close(fig2)

    for lvl_segs in CS.allsegs:
        for seg in lvl_segs:
            theta = seg[:,0]
            phi = seg[:,1]

            x = np.cos(phi)*np.sin(theta)
            y = np.sin(phi)*np.sin(theta)
            z = np.cos(theta)

            ax3d.plot(x, y, z, color=color, alpha=alpha, linewidth=linewidth)

def euclidean_kde_density(points, query_points, sigma=0.15):
    diff = query_points[:, None, :] - points[None, :, :]
    dist2 = np.sum(diff**2, axis=-1)
    density = np.exp(-dist2 / (2 * sigma**2)).mean(axis=1)
    return density


def add_density_contours_mesh(ax3d, mesh, points, levels=(0.4,0.6,0.8), sigma=0.15):
    vertices, faces = mesh

    density = euclidean_kde_density(points, vertices, sigma=sigma)
    density /= density.max()

    for lvl in levels:
        mask = np.abs(density - lvl) < 0.02

        ax3d.scatter(
            vertices[mask, 0],
            vertices[mask, 1],
            vertices[mask, 2],
            s=5,
            alpha=0.6
        )

    
def plot_object_points(mesh, points, point_colors, ax, show_contours=False, is_sphere=True):
    """
    Objective: Primary visualization tool for metric measure space alignment.
    Logic: Renders the object surface and overlays data-driven 
           density contours to visualize the probability measure signature.
    """
    vertices, faces = mesh

    ax.plot_trisurf(
        vertices[:,0], vertices[:,1], vertices[:,2],
        triangles=faces, cmap=blue_pink, alpha=0.1, edgecolor='none'
    )

    ax.scatter(
        points[:,0], points[:,1], points[:,2],
        c=point_colors, s=25, alpha=0.8,
        edgecolor='white', linewidth=0.3
    )

    if show_contours:
        if is_sphere:
            add_density_contours_s2(ax, points, color=point_colors)
        else:
            add_density_contours_mesh(ax, mesh, points)

    ax.set_box_aspect((1,1,1))
    ax.axis("off")


def plot_alignment_dashboard(src_data, tgt_data, theta_est, src_mesh, tgt_mesh, 
                             x_pair_src = None, x_pair_tgt = None, show_contours=False, is_sphere=True, view_angle=(-30, 0),
                             probe_points=None, mean_dir=None):
    """
    Objective: 3-panel dashboard to verify GW alignment.
    """

    R_final = get_rotation_matrix(theta_est)
    aligned_tgt = (R_final @ tgt_data.T).T

    fig = plt.figure(figsize=(18,6))

    ax1 = fig.add_subplot(131, projection='3d')
    plot_object_points(src_mesh, src_data, 'blue', ax1, show_contours, is_sphere)
    if x_pair_src is not None:
        ax1.scatter(x_pair_src[0], x_pair_src[1], x_pair_src[2], color='green', s=100)
    ax1.set_title("Source")
    ax1.view_init(*view_angle)

    ax2 = fig.add_subplot(132, projection='3d')
    plot_object_points(tgt_mesh, tgt_data, 'red', ax2, show_contours, is_sphere)
    if x_pair_tgt is not None:
        ax2.scatter(x_pair_tgt[0], x_pair_tgt[1], x_pair_tgt[2], color='purple', s=100)
    ax2.set_title("Target")
    ax2.view_init(*view_angle)

    ax3 = fig.add_subplot(133, projection='3d')
    plot_object_points(src_mesh, src_data, 'blue', ax3, show_contours, is_sphere)

    if x_pair_src is not None:
        ax3.scatter(x_pair_src[0], x_pair_src[1], x_pair_src[2], color='green', s=200)
    if x_pair_tgt is not None:
        x_pair_tgt_aligned = R_final @ x_pair_tgt
        ax3.scatter(
            x_pair_tgt_aligned[0],
            x_pair_tgt_aligned[1],
            x_pair_tgt_aligned[2],
            color='purple',
            s=100
        )
    ax3.view_init(*view_angle)

    ax3.scatter(
        aligned_tgt[:,0], aligned_tgt[:,1], aligned_tgt[:,2],
        color='red', s=20, alpha=0.8
    )

    if show_contours:
        add_density_contours_s2(ax3, aligned_tgt, color="red")

    ax3.set_title("Alignment Result")

    plt.tight_layout()
    plt.show()