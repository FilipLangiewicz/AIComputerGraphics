import cv2
import lpips
import torch

import numpy as np

from pathlib import Path
from tqdm import tqdm

from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from ddpm import DDPM
from unet import UNet


def compute_metrics(
        model: UNet,
        ddpm: DDPM,
        loader: DataLoader,
        device: str,
        num_samples: int = 32,
        save_images: bool = False,
        output_dir: Path = Path("outputs"),
        offset: int = 2400
) -> dict[str, float]:

    if save_images:
        generated_images_dir = output_dir / "generated_images"
        side_by_side_dir = output_dir / "side_by_side"

        output_dir.mkdir(exist_ok=True, parents=True)
        generated_images_dir.mkdir(exist_ok=True, parents=True)
        side_by_side_dir.mkdir(exist_ok=True, parents=True)

    lpips_fn = lpips.LPIPS(net='alex').to(device)
    results = {"ssim": [], "lpips": [], "hausdorff": []}
    sample_count = 0

    with torch.no_grad():
        for params, real_images in tqdm(loader, desc="Metrics", leave=False, total=num_samples):
            params = params.to(device)
            real_images = real_images.to(device)

            B = params.size(0)

            generated_images = ddpm.ddim_sample_loop(model, params, (B, 3, 128, 128), ddim_steps=50)

            lp = lpips_fn(generated_images.clamp(-1, 1), real_images.clamp(-1, 1))
            results["lpips"].extend(lp.squeeze().cpu().tolist() if lp.numel() > 1 else [lp.item()])

            for i in range(B):
                ref_np = tensor_to_np(real_images[i])
                gen_np = tensor_to_np(generated_images[i])

                results["ssim"].append(ssim(ref_np, gen_np, data_range=1.0, channel_axis=2))
                results["hausdorff"].append(compute_hausdorff(ref_np, gen_np))

                if save_images:
                    save_image(generated_images[i].clamp(-1, 1), generated_images_dir / f"image_{sample_count + offset:03d}.png")

                    side_by_side = torch.cat([real_images[i], generated_images[i].clamp(-1, 1)], dim=2)
                    save_image((side_by_side + 1) / 2, side_by_side_dir / f"image_{sample_count + offset:03d}.png")

                sample_count += 1

            if sample_count >= num_samples:
                break

    return {
        "SSIM": float(np.nanmean(results["ssim"])),
        "LPIPS": float(np.nanmean(results["lpips"])),
        "Hausdorff": float(np.nanmean([x for x in results["hausdorff"] if not np.isnan(x)])),
    }


def compute_hausdorff(img_ref, img_gen, max_points=500):
    gray_ref = cv2.cvtColor((img_ref * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_gen = cv2.cvtColor((img_gen * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    edges_ref = cv2.Canny(gray_ref, 100, 200)
    edges_gen = cv2.Canny(gray_gen, 100, 200)

    pts_ref = np.column_stack(np.where(edges_ref > 0))
    pts_gen = np.column_stack(np.where(edges_gen > 0))

    if len(pts_ref) == 0 or len(pts_gen) == 0:
        return float('nan')

    if len(pts_ref) > max_points:
        pts_ref = pts_ref[np.random.choice(len(pts_ref), max_points, replace=False)]

    if len(pts_gen) > max_points:
        pts_gen = pts_gen[np.random.choice(len(pts_gen), max_points, replace=False)]

    return max(directed_hausdorff(pts_ref, pts_gen)[0], directed_hausdorff(pts_gen, pts_ref)[0])


def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    return ((t.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()
