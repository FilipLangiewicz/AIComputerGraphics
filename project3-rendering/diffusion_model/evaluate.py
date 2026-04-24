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


def evaluate(model: UNet, ddpm: DDPM, test_loader: DataLoader, device: str, output_dir: Path, num_samples: int = 600, offset: int = 2400):
    generated_images_dir = output_dir / "generated_images"
    side_by_side_dir = output_dir / "side_by_side"

    output_dir.mkdir(exist_ok=True, parents=True)
    generated_images_dir.mkdir(exist_ok=True, parents=True)
    side_by_side_dir.mkdir(exist_ok=True, parents=True)

    lpips_fn = lpips.LPIPS(net='alex').to(device)

    results = {"ssim": [], "lpips": [], "hausdorff": []}
    sample_count = 0

    for params, real_images in tqdm(test_loader, desc="Evaluating"):
        params = params.to(device)
        real_images = real_images.to(device)

        B = params.size(0)

        generated_images = ddpm.p_sample_loop(model, params, (B, 3, 128, 128))

        # LPIPS
        lp = lpips_fn(generated_images.clamp(-1, 1), real_images.clamp(-1, 1))
        results["lpips"].extend(lp.squeeze().cpu().tolist() if lp.numel() > 1 else [lp.item()])

        for i in range(B):
            ref_np = tensor_to_np(real_images[i])
            gen_np = tensor_to_np(generated_images[i])

            # SSIM
            s = ssim(ref_np, gen_np, data_range=1.0, channel_axis=2)
            results["ssim"].append(s)

            # Hausdorff na krawędziach Canny
            h = compute_hausdorff(ref_np, gen_np)
            results["hausdorff"].append(h)

            # Zapisz wygenerowane obrazy
            save_image(generated_images[i].clamp(-1, 1), generated_images_dir / f"image_{sample_count+offset:03d}.png")

            side_by_side = torch.cat([real_images[i], generated_images[i].clamp(-1, 1)], dim=2)
            save_image((side_by_side + 1) / 2, side_by_side_dir / f"image_{sample_count+offset:03d}.png")

            sample_count += 1

        if sample_count >= num_samples:
            break

    metrics = {
        "SSIM": float(np.nanmean(results["ssim"])),
        "LPIPS": float(np.nanmean(results["lpips"])),
        "Hausdorff": float(np.nanmean([x for x in results["hausdorff"] if not np.isnan(x)])),
    }

    return metrics


def compute_hausdorff(img_ref: np.ndarray, img_gen: np.ndarray) -> float:
    gray_ref = cv2.cvtColor((img_ref * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_gen = cv2.cvtColor((img_gen * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    edges_ref = cv2.Canny(gray_ref, 100, 200)
    edges_gen = cv2.Canny(gray_gen, 100, 200)

    pts_ref = np.column_stack(np.where(edges_ref > 0))
    pts_gen = np.column_stack(np.where(edges_gen > 0))

    if len(pts_ref) == 0 or len(pts_gen) == 0:
        return float('nan')

    d1 = directed_hausdorff(pts_ref, pts_gen)[0]
    d2 = directed_hausdorff(pts_gen, pts_ref)[0]

    return max(d1, d2)


def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    return ((t.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()
