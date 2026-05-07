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
    
def visualize_transition_3d(
    step_clouds: list[np.ndarray],
    titles: list[str] = None,
    point_size: int = 2,
):
    """Interactive 3D visualization of transition steps using plotly."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n = len(step_clouds)
    fig = make_subplots(
        rows=1, cols=n,
        specs=[[{"type": "scatter3d"}] * n],
        subplot_titles=titles or [f"Step {i}" for i in range(n)],
    )

    for i, cloud in enumerate(step_clouds):
        fig.add_trace(
            go.Scatter3d(
                x=cloud[:, 0], y=cloud[:, 1], z=cloud[:, 2],
                mode="markers",
                marker=dict(size=point_size, color=cloud[:, 2], colorscale="Viridis"),
                showlegend=False,
            ),
            row=1, col=i + 1,
        )

    fig.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0))
    fig.show()