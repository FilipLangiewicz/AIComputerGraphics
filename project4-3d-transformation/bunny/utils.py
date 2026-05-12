from pathlib import Path

import trimesh

import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    return trimesh.load(path, force="mesh")


def load_mesh_pointcloud(path: str | Path, n_points: int) -> np.ndarray:
    mesh = load_mesh(path)
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)

    return pts.astype(np.float32)


def pointcloud_to_mesh(points: np.ndarray) -> trimesh.Trimesh:
    try:
        pc = trimesh.PointCloud(points)

        return pc.convex_hull
    except Exception:
        return trimesh.convex.convex_hull(points)


def standardize_pointcloud(points: np.ndarray) -> np.ndarray:
    centroid = points.mean(axis=0)
    pts = points - centroid
    max_dist = np.linalg.norm(pts, axis=1).max()

    return (pts / max_dist).astype(np.float32)


def visualize_transition(
    step_clouds: list[np.ndarray],
    titles: list[str] = None,
    point_size: float = 0.5,
    figsize: tuple = (18, 5),
):
    num_steps = len(step_clouds)
    fig = plt.figure(figsize=figsize)

    for idx, cloud in enumerate(step_clouds):
        ax = fig.add_subplot(1, num_steps, idx + 1, projection="3d")
        ax.scatter(
            cloud[:, 0], cloud[:, 1], cloud[:, 2],
            s=point_size, c=cloud[:, 2], cmap="viridis"
        )
        ax.set_axis_off()
        if titles:
            ax.set_title(titles[idx])

    plt.tight_layout()
    plt.show()


def interpolate_pointclouds(src: np.ndarray, tgt: np.ndarray, steps: int = 3) -> list[np.ndarray]:
    return [
        src + (tgt - src) * t
        for t in np.linspace(0, 1, steps)
    ]


def visualize_transition_3d(
    step_clouds: list[np.ndarray],
    titles: list[str] = None,
    point_size: int = 2,
):
    num_steps = len(step_clouds)

    fig = make_subplots(
        rows=1, cols=num_steps,
        specs=[[{"type": "scatter3d"}] * num_steps],
        subplot_titles=titles or [f"Step {i}" for i in range(num_steps)],
    )

    for idx, cloud in enumerate(step_clouds):
        fig.add_trace(
            go.Scatter3d(
                x=cloud[:, 0], y=cloud[:, 1], z=cloud[:, 2],
                mode="markers",
                marker=dict(size=point_size, color=cloud[:, 2], colorscale="Viridis"),
                showlegend=False,
            ),
            row=1, col=idx + 1,
        )

    fig.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0))
    fig.show()