import torch
import numpy as np
from torch.utils.data import Dataset
import trimesh


def load_obj_pointcloud(path: str, n_points: int) -> np.ndarray:
    """Sample n_points uniformly from mesh surface. Returns (N, 3) array."""
    mesh = trimesh.load(path, force="mesh")
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return points.astype(np.float32)


def normalize_pointcloud(points: np.ndarray) -> np.ndarray:
    """Center and scale to unit sphere."""
    points = points - points.mean(axis=0)
    scale = np.linalg.norm(points, axis=1).max()
    return points / scale


def random_rotation_matrix() -> np.ndarray:
    """Uniform random rotation in SO(3)."""
    q, _ = np.linalg.qr(np.random.randn(3, 3))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float32)


def random_scale(scale_range: tuple[float, float] = (0.75, 1.25)) -> float:
    return np.random.uniform(*scale_range)


class ArmadilloTeapotDataset(Dataset):
    """
    Each sample: randomly augmented armadillo point cloud -> normalized teapot point cloud.

    Args:
        armadillo_path: path to armadillo .obj
        teapot_path:    path to teapot .obj
        n_points:       number of points sampled per mesh
        n_samples:      virtual dataset size (samples generated on-the-fly)
        augment:        whether to apply random rotation + scale to input
        scale_range:    min/max scale factor for augmentation
    """

    def __init__(
        self,
        armadillo_path: str,
        teapot_path: str,
        n_points: int = 2048,
        n_samples: int = 10000,
        augment: bool = True,
        scale_range: tuple[float, float] = (0.75, 1.25),
    ):
        self.n_points = n_points
        self.n_samples = n_samples
        self.augment = augment
        self.scale_range = scale_range

        # load and normalize both meshes once
        self.armadillo_mesh = trimesh.load(armadillo_path, force="mesh")
        self.teapot_points = normalize_pointcloud(
            load_obj_pointcloud(teapot_path, n_points)
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # fresh surface sample every call -> variability across epochs
        pts, _ = trimesh.sample.sample_surface(self.armadillo_mesh, self.n_points)
        pts = normalize_pointcloud(pts.astype(np.float32))

        if self.augment:
            R = random_rotation_matrix()
            s = random_scale(self.scale_range)
            pts = (pts @ R.T) * s

        return torch.from_numpy(pts), torch.from_numpy(self.teapot_points.copy())