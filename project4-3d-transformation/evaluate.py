import numpy as np
import trimesh
from scipy.spatial import cKDTree


def _mesh_to_voxels(mesh: trimesh.Trimesh, resolution: int = 64) -> np.ndarray:
    """Voxelize a mesh into a boolean 3D grid of given resolution."""
    bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    pitch = (bounds[1] - bounds[0]).max() / resolution
    voxels = mesh.voxelized(pitch=pitch).fill()
    return voxels.matrix.astype(bool)


def _align_voxel_grids(grid_a: np.ndarray, grid_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pad both grids to the same shape for elementwise comparison."""
    max_shape = np.maximum(grid_a.shape, grid_b.shape)
    def pad(g, s):
        pad_width = [(0, s[i] - g.shape[i]) for i in range(3)]
        return np.pad(g, pad_width)
    return pad(grid_a, max_shape), pad(grid_b, max_shape)


def iou(mesh_pred: trimesh.Trimesh, mesh_target: trimesh.Trimesh, resolution: int = 64) -> float:
    """Intersection over Union (voxel-based)."""
    a, b = _align_voxel_grids(
        _mesh_to_voxels(mesh_pred, resolution),
        _mesh_to_voxels(mesh_target, resolution)
    )
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union > 0 else 0.0


def dice(mesh_pred: trimesh.Trimesh, mesh_target: trimesh.Trimesh, resolution: int = 64) -> float:
    """Dice coefficient (voxel-based)."""
    a, b = _align_voxel_grids(
        _mesh_to_voxels(mesh_pred, resolution),
        _mesh_to_voxels(mesh_target, resolution)
    )
    intersection = np.logical_and(a, b).sum()
    return float(2 * intersection / (a.sum() + b.sum())) if (a.sum() + b.sum()) > 0 else 0.0


def chamfer_distance(points_pred: np.ndarray, points_target: np.ndarray) -> float:
    """
    Chamfer distance between two point clouds.
    
    Args:
        points_pred:   (N, 3) array of predicted points
        points_target: (M, 3) array of target points
    """
    tree_target = cKDTree(points_target)
    tree_pred = cKDTree(points_pred)

    dist_pred_to_target, _ = tree_target.query(points_pred, k=1)
    dist_target_to_pred, _ = tree_pred.query(points_target, k=1)

    return float(np.mean(dist_pred_to_target**2) + np.mean(dist_target_to_pred**2))


def evaluate_all(
    mesh_pred: trimesh.Trimesh,
    mesh_target: trimesh.Trimesh,
    n_points: int = 10000,
    resolution: int = 64
) -> dict:
    """
    Compute all three metrics at once.
    
    Returns dict with keys: 'iou', 'dice', 'chamfer'
    """
    points_pred = trimesh.sample.sample_surface(mesh_pred, n_points)[0]
    points_target = trimesh.sample.sample_surface(mesh_target, n_points)[0]

    return {
        "iou": iou(mesh_pred, mesh_target, resolution),
        "dice": dice(mesh_pred, mesh_target, resolution),
        "chamfer": chamfer_distance(points_pred, points_target),
    }