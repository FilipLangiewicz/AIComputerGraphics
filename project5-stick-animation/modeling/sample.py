from pathlib import Path

import torch
import matplotlib.pyplot as plt

from model import MotionDenoiser
from diffusion import GaussianDiffusion
from train import load_model
from utils import animate_skeleton_3d


CLASS_NAMES = {0: "walk", 1: "jump"}


def generate(
    checkpoint_path: str,
    class_label: int,
    n_samples: int = 4,
    guidance_scale: float = 3.0,
    n_joints: int = 15,
    n_frames: int = 48,
    d_model: int = 256,
    nhead: int = 4,
    num_layers: int = 4,
    num_classes: int = 2,
    dropout: float = 0.0,
    timesteps: int = 1000,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate motion samples. Returns [n_samples, n_frames, n_joints, 3]."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, n_joints, n_frames, d_model,
                       nhead, num_layers, num_classes, dropout, str(device))
    diffusion = GaussianDiffusion(timesteps=timesteps).to(device)
    labels = torch.full((n_samples,), class_label, dtype=torch.long, device=device)
    with torch.no_grad():
        samples = diffusion.sample(model, labels, n_frames=n_frames,
                                   n_joints=n_joints, guidance_scale=guidance_scale)
    return samples.cpu()


def save_samples(samples: torch.Tensor, class_label: int,
                 save_dir: str = "generated", fps: int = 24) -> None:
    """Save each sample as GIF."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    name = CLASS_NAMES.get(class_label, f"class{class_label}")
    for i, motion in enumerate(samples):
        out = str(save_dir / f"{name}_s{i+1}.gif")
        animate_skeleton_3d(motion.numpy(), output_filename=out, fps=fps)
        plt.close()
        print(f"saved → {out}")


def visualize_training_samples(samples_pt: str, save_dir: str = None,
                                fps: int = 24) -> None:
    """Visualize samples_e*.pt saved during training."""
    data = torch.load(samples_pt)
    save_dir = Path(save_dir) if save_dir else Path(samples_pt).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    for cls_id, motions in data.items():
        name = CLASS_NAMES.get(cls_id, f"class{cls_id}")
        for i, motion in enumerate(motions):
            out = str(save_dir / f"{name}_s{i+1}.gif")
            animate_skeleton_3d(motion.numpy(), output_filename=out, fps=fps)
            plt.close()
            print(f"saved → {out}")