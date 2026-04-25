from pathlib import Path

import numpy as np
import torch
import cv2
from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import structural_similarity as ssim_fn
import lpips
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

def to_numpy_uint8(t: torch.Tensor) -> np.ndarray:
    """Tensor (C,H,W) in [-1,1] -> numpy (H,W,C) uint8."""
    arr = t.detach().cpu().permute(1, 2, 0).numpy()
    return ((arr * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)


def hausdorff_canny(
    img_a: np.ndarray,
    img_b: np.ndarray,
    low:   int = 50,
    high:  int = 150,
) -> float:
    """Symmetric Hausdorff distance on Canny edge maps."""
    gray_a  = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    gray_b  = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
    edges_a = cv2.Canny(gray_a, low, high)
    edges_b = cv2.Canny(gray_b, low, high)
    pts_a   = np.argwhere(edges_a)
    pts_b   = np.argwhere(edges_b)
    if pts_a.size == 0 or pts_b.size == 0:
        return 0.0
    return max(directed_hausdorff(pts_a, pts_b)[0],
               directed_hausdorff(pts_b, pts_a)[0])


def evaluate(
    G,
    test_ds,
    output_dir:  Path,
    ckpt_path:   Path = None,
    batch_size:  int  = 32,
    num_workers: int  = 2,
    device:      str  = "cuda",
) -> dict:
    """Run FLIP / LPIPS / SSIM / Hausdorff on the test set."""

    pin_memory  = (device == "cuda")
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=pin_memory)
    output_dir = Path(output_dir)

    if ckpt_path:
        G.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded checkpoint: {ckpt_path}")
    G.eval()

    noise_dim = G.noise_dim
    lpips_fn  = lpips.LPIPS(net="alex").to(device)

    try:
        import flip as flip_lib
        HAS_FLIP = True
        print("FLIP library available")
    except ImportError:
        HAS_FLIP = False
        print("flip-evaluator not found, FLIP will be N/A")

    all_lpips, all_ssim, all_hausdorff, all_flip = [], [], [], []

    with torch.no_grad():
        for real_imgs, cond in tqdm(test_loader, desc="Evaluating", ncols=80):
            real_imgs = real_imgs.to(device)
            cond      = cond.to(device)
            B         = real_imgs.size(0)
            z         = torch.zeros(B, noise_dim, device=device)
            fake_imgs = G(z, cond)

            lp_vals = lpips_fn(fake_imgs, real_imgs).squeeze().cpu().numpy()
            all_lpips.extend(np.atleast_1d(lp_vals).tolist())

            for j in range(B):
                real_np = to_numpy_uint8(real_imgs[j])
                fake_np = to_numpy_uint8(fake_imgs[j])
                all_ssim.append(ssim_fn(real_np, fake_np, channel_axis=2, data_range=255))
                all_hausdorff.append(hausdorff_canny(real_np, fake_np))
                if HAS_FLIP:
                    r_f = real_np.astype(np.float32) / 255.0
                    f_f = fake_np.astype(np.float32) / 255.0
                    all_flip.append(float(flip_lib.compute_flip(r_f, f_f).mean()))

    flip_mean = f"{np.mean(all_flip):.4f}" if HAS_FLIP else "  N/A  "

    print("\n" + "=" * 60)
    print(f"{'Method':<22} {'FLIP':>8} {'LPIPS':>8} {'SSIM':>8} {'Hausdorff':>10}")
    print("-" * 60)
    print(f"{'neural_renderer':<22} {flip_mean:>8} "
          f"{np.mean(all_lpips):>8.4f} "
          f"{np.mean(all_ssim):>8.4f} "
          f"{np.mean(all_hausdorff):>10.2f}")
    print("=" * 60)

    results = {
        "FLIP":      float(np.mean(all_flip)) if HAS_FLIP else None,
        "LPIPS":     float(np.mean(all_lpips)),
        "SSIM":      float(np.mean(all_ssim)),
        "Hausdorff": float(np.mean(all_hausdorff)),
    }

    with open(output_dir / "metrics.txt", "w") as f:
        f.write(f"FLIP      : {flip_mean}\n")
        f.write(f"LPIPS     : {results['LPIPS']:.4f}\n")
        f.write(f"SSIM      : {results['SSIM']:.4f}\n")
        f.write(f"Hausdorff : {results['Hausdorff']:.2f}\n")

    print(f"Metrics saved to: {output_dir / 'metrics.txt'}")
    return results


def visualize_results(
    G,
    test_ds,
    output_dir: Path,
    device:     str = "cuda",
    batch_size:  int  = 32,
    num_workers: int  = 2,
    n_cols:     int = 8,
    n_pairs:    int = 2,
):
    """Grid: reference (odd rows) vs generated (even rows)."""
    
    pin_memory  = (device == "cuda")
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=pin_memory)
    
    G.eval()
    output_dir = Path(output_dir)
    noise_dim  = G.noise_dim

    n = n_cols * n_pairs
    all_real, all_fake = [], []

    with torch.no_grad():
        for real_imgs, cond in test_loader:
            real_imgs = real_imgs.to(device)
            cond      = cond.to(device)
            z         = torch.zeros(real_imgs.size(0), noise_dim, device=device)
            all_real.append(real_imgs.cpu())
            all_fake.append(G(z, cond).cpu())
            if sum(r.size(0) for r in all_real) >= n:
                break

    real_cat = torch.cat(all_real)[:n]
    fake_cat = torch.cat(all_fake)[:n]

    fig, axes = plt.subplots(n_pairs * 2, n_cols, figsize=(n_cols * 2, n_pairs * 4))
    fig.suptitle("Reference vs. Neural Renderer", fontsize=13)

    for pair in range(n_pairs):
        for col in range(n_cols):
            idx = pair * n_cols + col
            axes[pair*2,   col].imshow(to_numpy_uint8(real_cat[idx]))
            axes[pair*2,   col].axis("off")
            axes[pair*2+1, col].imshow(to_numpy_uint8(fake_cat[idx]))
            axes[pair*2+1, col].axis("off")
        axes[pair*2,   0].set_ylabel("Real",      fontsize=11)
        axes[pair*2+1, 0].set_ylabel("Generated", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / "results_comparison.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    from pathlib import Path
    from dataset import get_loaders
    from models import build_models
    from config import DEVICE

    IMAGES_DIR = Path("../data/images")
    PARAMS_DIR = Path("../data/params")
    OUTPUT_DIR = Path("output")
    CKPT_DIR   = Path("checkpoints")

    _, test_loader = get_loaders(IMAGES_DIR, PARAMS_DIR)
    G, _ = build_models(device=DEVICE)
    ckpt = sorted(CKPT_DIR.glob("G_*.pth"))[-1]
    evaluate(G, test_loader, OUTPUT_DIR, ckpt_path=ckpt, device=DEVICE)
    visualize_results(G, test_loader, OUTPUT_DIR, device=DEVICE)