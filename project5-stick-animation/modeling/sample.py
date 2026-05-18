from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

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


def save_samples(
    samples: torch.Tensor,
    class_label: int,
    save_dir: str = "generated",
    fps: int = 24,
    show_plots: bool = False,
) -> None:
    """Save each sample as GIF."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    name = CLASS_NAMES.get(class_label, f"class{class_label}")
    for i, motion in enumerate(samples):
        out = str(save_dir / f"{name}_s{i+1}.gif")
        animate_skeleton_3d(
            motion.numpy(),
            output_filename=out,
            fps=fps,
            show=show_plots,
        )
        plt.close()
        print(f"saved → {out}")


def visualize_training_samples(
    samples_pt: str,
    save_dir: str = None,
    fps: int = 24,
    show_plots: bool = False,
) -> None:
    """Visualize samples_e*.pt saved during training."""
    data = torch.load(samples_pt)
    save_dir = Path(save_dir) if save_dir else Path(samples_pt).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    for cls_id, motions in data.items():
        name = CLASS_NAMES.get(cls_id, f"class{cls_id}")
        for i, motion in enumerate(motions):
            out = str(save_dir / f"{name}_s{i+1}.gif")
            animate_skeleton_3d(
                motion.numpy(),
                output_filename=out,
                fps=fps,
                show=show_plots,
            )
            plt.close()
            print(f"saved → {out}")
            
            
def denormalize_samples(samples: torch.Tensor, norm_stats_path: str) -> torch.Tensor:
    """
    Denormalize samples using mean and std stored in a numpy file.
    Expected stats shape: [2, ...] where stats[0] = mean and stats[1] = std.
    """
    stats_path = Path(norm_stats_path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {stats_path}")

    stats = np.load(stats_path)
    mean = torch.tensor(stats[0], dtype=samples.dtype, device=samples.device)
    std = torch.tensor(stats[1], dtype=samples.dtype, device=samples.device)
    return samples * std + mean


def save_sample_as_gif(
    sample: torch.Tensor,
    class_label: int,
    sample_index: int,
    save_dir: str,
    fps: int = 24,
    show_plots: bool = False,
) -> Path:
    """
    Save one sample as a GIF and return the output path.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    class_name = CLASS_NAMES.get(class_label, f"class{class_label}")
    output_path = save_dir / f"{class_name}_s{sample_index + 1}.gif"

    animate_skeleton_3d(
        sample.cpu().numpy(),
        output_filename=str(output_path),
        fps=fps,
        show=show_plots,
    )
    return output_path


def save_samples_as_gifs(
    samples: torch.Tensor,
    class_label: int,
    save_dir: str,
    fps: int = 24,
    show_plots: bool = False,
) -> list[Path]:
    """
    Save all samples from a batch as GIFs and return output paths.
    Expected shape: [n_samples, n_frames, n_joints, 3].
    """
    output_paths = []
    for i, sample in enumerate(samples):
        output_path = save_sample_as_gif(
            sample=sample,
            class_label=class_label,
            sample_index=i,
            save_dir=save_dir,
            fps=fps,
            show_plots=show_plots,
        )
        output_paths.append(output_path)
    return output_paths


def generate_denormalize_animate_and_save(
    checkpoint_path: str,
    class_label: int,
    save_dir: str,
    n_samples: int = 10,
    guidance_scale: float = 3.0,
    n_joints: int = 15,
    n_frames: int = 48,
    d_model: int = 256,
    nhead: int = 4,
    num_layers: int = 4,
    num_classes: int = 2,
    dropout: float = 0.0,
    timesteps: int = 1000,
    fps: int = 24,
    norm_stats_path: str = None,
    device: str = "cuda",
    display_gifs: bool = False,
    show_plots: bool = False,
):
    samples = generate(
        checkpoint_path=checkpoint_path,
        class_label=class_label,
        n_samples=n_samples,
        guidance_scale=guidance_scale,
        n_joints=n_joints,
        n_frames=n_frames,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        timesteps=timesteps,
        device=device,
    )

    if norm_stats_path is not None:
        samples = denormalize_samples(samples, norm_stats_path)
    
    gif_paths = save_samples_as_gifs(
        samples=samples,
        class_label=class_label,
        save_dir=save_dir,
        fps=fps,
        show_plots=show_plots,
    )

    if display_gifs:
        from IPython.display import Image, display, Markdown

        class_name = CLASS_NAMES.get(class_label, f"class{class_label}")
        display(Markdown(f"## {class_name} GIFs"))
        for gif_path in gif_paths:
            display(Image(filename=str(gif_path)))

    return samples, gif_paths