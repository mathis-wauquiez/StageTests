

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

import torch

from IPython.display import HTML
from torch.utils.data import DataLoader

from scipy.stats import gaussian_kde


def scatter_points(x0, x1):
    plt.figure(figsize=(6, 6))
    plt.scatter(x0[:, 0], x0[:, 1], color='blue', alpha=0.3, label='Source')
    plt.scatter(x1[:, 0], x1[:, 1], color='orange', alpha=0.3, label='Target')
    plt.legend()
    plt.axis('equal')
    plt.title('Samples')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

def scatter_points_with_velocity(x0, x1, x_t, target_vectors):
    plt.figure(figsize=(6, 6))
    plt.scatter(x0[:, 0], x0[:, 1], color='blue', alpha=0.3, label='Source (GMM1)')
    plt.scatter(x1[:, 0], x1[:, 1], color='orange', alpha=0.3, label='Target (GMM2)')
    plt.scatter(x_t[:, 0], x_t[:, 1], color='red', alpha=0.3, label='Transformed samples')
    plt.quiver(x_t[:, 0], x_t[:, 1], target_vectors[:, 0], target_vectors[:, 1], color='green', alpha=0.5, label='velocity')
    plt.legend()
    plt.axis('equal')
    plt.title('Samples from GMM2GMM dataset with transformed samples and target velocity')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()


def animate_estimated_velocity(x0, x1, estimated_velocity, device="cuda"):
    # === Grid ===
    grid_size = 20
    x_vals = np.linspace(-3, 3, grid_size)
    y_vals = np.linspace(-3, 3, grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)
    XY = np.stack([X.ravel(), Y.ravel()], axis=1)  # shape (grid_size^2, 2)
    XY_torch = torch.tensor(XY, dtype=torch.float32, device=device)  # (N, 2)

    # === Time steps ===
    time_steps = torch.linspace(0, 1, 100, device=device)

    # === Figure setup ===
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.scatter(x0[:, 0], x0[:, 1], color='blue', alpha=0.3, label='Source (GMM1)')

    plt.scatter(x1[:, 0], x1[:, 1], color='orange', alpha=0.3, label='Target (GMM2)')

    quiver = ax.quiver(X, Y, np.zeros_like(X), np.zeros_like(Y))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_title("Estimated Velocity Field over Time")

    # === Animation function ===
    def update(frame):
        t = time_steps[frame]
        t_tensor = t.expand(XY_torch.shape[0])  # (N,)
        v = estimated_velocity(t_tensor, XY_torch)  # (N, 2)
        v_cpu = v.detach().cpu().numpy()
        U = v_cpu[:, 0].reshape(grid_size, grid_size)
        V = v_cpu[:, 1].reshape(grid_size, grid_size)
        quiver.set_UVC(U, V)
        ax.set_title(f"Velocity Field at t = {t.item():.2f}")
        return quiver,

    ani = FuncAnimation(fig, update, frames=100, interval=50)
    plt.close(fig)  # Prevents double display in notebooks
    return ani



def animate_sampled_trajectories(
    flow_model,
    dataset,
    samples=500,
    y=None
):
    """
    Animates trajectories returned by `flow_model.sample_trajectory(x0)`.

    Args:
        flow_model: Object with method `sample_trajectory(x0)` returning (trajectories, t)
        dataset: Dataset returning (x0, x1) pairs.
        samples: Number of particles to animate.
    """
    N = samples
    loader = DataLoader(dataset, batch_size=samples, shuffle=True)
    x0, _ , _= next(iter(loader))

    if y is not None:
        # y is a int that we need to expand to the batch size
        y = torch.full((N,), y, device=x0.device)

    # Get trajectories and time points
    trajectories, t = flow_model.sample_trajectory(x0, y=y)  # (n_steps, N, D), (n_steps,)
    trajectories = trajectories.detach().cpu().numpy()

    fig, ax = plt.subplots()
    lines = [ax.plot([], [], 'o')[0] for _ in range(N)]

    def init():
        ax.set_xlim(trajectories[:, :, 0].min() - 1, trajectories[:, :, 0].max() + 1)
        ax.set_ylim(trajectories[:, :, 1].min() - 1, trajectories[:, :, 1].max() + 1)
        return lines

    def animate(i):
        for j, line in enumerate(lines):
            x, y = trajectories[i, j, 0], trajectories[i, j, 1]
            line.set_data([x], [y])
        return lines

    ani = animation.FuncAnimation(
        fig, animate, frames=len(trajectories), init_func=init, blit=True, interval=40
    )

    plt.close(fig)
    return HTML(ani.to_jshtml())


def plot_trajectories_with_density(
    flow_model,
    dataset,
    *,
    N=1_000,                 # nombre de particules visualisées
    # --- style du subplot 1 (trajectoires) ----
    path_color="royalblue",
    path_alpha=0.10,
    linewidth=0.75,
    start_color="purple",
    end_color="crimson",
    scatter_size=12,
    # --- style du subplot 2 (densité) ---------
    grid_size=200,           # résolution du maillage KDE
    cmap="viridis",
    figsize=(6, 10),
):
    """
    Trace (1) les trajectoires brutes   (2) une estimation de densité 2-D.

    flow_model.sample_trajectory(x0) doit retourner:
        traj  : (n_steps, N, 2)
        t_grid: (n_steps,)
    """
    # 1) Charger un batch
    loader = DataLoader(dataset, batch_size=N, shuffle=True)
    x0, x1, _ = next(iter(loader))
    x0 = x0.to(flow_model.device)  # (N, 2)
    x1 = x1.to(flow_model.device)  # (N, 2)

    # 2) Échantillonner les trajectoires
    traj, t_grid = flow_model.sample_trajectory(x0)         # (T, N, 2)
    traj = traj.detach().cpu().numpy()                      # -> numpy
    n_steps = traj.shape[0]

    # 3) Préparer figure
    fig, (ax_traj, ax_kde) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 2]}
    )

    # ---------- Subplot 1 : trajectoires -------------------
    for j in range(N):
        ax_traj.plot(
            traj[:, j, 0], traj[:, j, 1],
            color=path_color, alpha=path_alpha, linewidth=linewidth
        )
    # points début / fin
    ax_traj.scatter(traj[0, :, 0], traj[0, :, 1],
                    c=start_color, s=scatter_size, label=r"$x_0$")
    ax_traj.scatter(traj[-1, :, 0], traj[-1, :, 1],
                    c=end_color,   s=scatter_size, label=r"$x_1$")

    ax_traj.set_aspect("equal", "box")
    ax_traj.axis("off")
    ax_traj.legend(loc="upper right", frameon=False)

    # ---------- Subplot 2 : carte de densité ---------------
    # Aplatir tous les points (T × N, 2)
    flat_pts = traj.reshape(-1, 2).T                        # shape (2, M)
    # KDE 2-D (gaussian)
    kde = gaussian_kde(flat_pts)
    # Maillage régulier
    x_min, x_max = flat_pts[0].min(), flat_pts[0].max()
    y_min, y_max = flat_pts[1].min(), flat_pts[1].max()
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size)
    )
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    im = ax_kde.imshow(
        zz, origin="lower", extent=[x_min, x_max, y_min, y_max],
        cmap=cmap, aspect="equal"
    )
    cbar = fig.colorbar(im, ax=ax_kde, fraction=0.046, pad=0.04)
    cbar.set_label("Estimated density")

    ax_kde.set_xlabel(r"$x$");  ax_kde.set_ylabel(r"$y$")
    ax_kde.set_title("Densité 2-D des points $x_t$ le long des trajectoires")

    plt.tight_layout()
    plt.show()
    return fig
