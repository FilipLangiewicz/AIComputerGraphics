import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_mesh(path: str) -> trimesh.Trimesh:
    return trimesh.load(path, force="mesh")


def pointcloud_to_mesh(points: np.ndarray) -> trimesh.Trimesh:
    """Reconstruct mesh from point cloud using ball-pivot (convex hull fallback)."""
    try:
        cloud = trimesh.PointCloud(points)
        return cloud.convex_hull
    except Exception:
        return trimesh.convex.convex_hull(points)


def visualize_transition(
    step_clouds: list[np.ndarray],
    titles: list[str] = None,
    point_size: float = 0.5,
    figsize: tuple = (18, 5),
):
    """
    Visualize a list of point clouds as 3D scatter plots side-by-side.
    
    Args:
        step_clouds: list of (N, 3) arrays — each is one transition step
        titles:      optional list of subplot titles
        point_size:  scatter dot size
    """
    n = len(step_clouds)
    fig = plt.figure(figsize=figsize)

    for i, cloud in enumerate(step_clouds):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=point_size, c=cloud[:, 2], cmap="viridis")
        ax.set_axis_off()
        if titles:
            ax.set_title(titles[i])

    plt.tight_layout()
    plt.show()


def interpolate_pointclouds(src: np.ndarray, tgt: np.ndarray, steps: int = 3) -> list[np.ndarray]:
    """
    Linear interpolation between two point clouds.
    Returns list of `steps` intermediate point clouds (including src and tgt).
    """
    return [
        src + (tgt - src) * t
        for t in np.linspace(0, 1, steps)
    ]