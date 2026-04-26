from pathlib import Path

import cv2
import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import structural_similarity as ssim_fn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm



def to_numpy_uint8(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().permute(1, 2, 0).numpy()
    return ((arr * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)


def hausdorff_canny(img_a: np.ndarray, img_b: np.ndarray, low: int = 50, high: int = 150) -> float:
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
    edges_a = cv2.Canny(gray_a, low, high)
    edges_b = cv2.Canny(gray_b, low, high)
    pts_a = np.argwhere(edges_a)
    pts_b = np.argwhere(edges_b)
    if pts_a.size == 0 or pts_b.size == 0:
        return 0.0
    return max(directed_hausdorff(pts_a, pts_b)[0], directed_hausdorff(pts_b, pts_a)[0])


def save_test_images(G, test_ds, save_dir: Path, device: str = "cuda",
                     batch_size: int = 32, num_workers: int = 2):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pin_memory = (device == "cuda")
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)
    G.eval()
    noise_dim = G.noise_dim
    idx = 0

    with torch.no_grad():
        for real_imgs, cond in tqdm(loader, desc="Saving images", ncols=80):
            real_imgs = real_imgs.to(device)
            cond = cond.to(device)
            z = torch.zeros(real_imgs.size(0), noise_dim, device=device)
            fake_imgs = G(z, cond)

            for j in range(real_imgs.size(0)):
                save_image(fake_imgs[j] * 0.5 + 0.5, save_dir / f"{idx:04d}_fake.png")
                save_image(real_imgs[j] * 0.5 + 0.5, save_dir / f"{idx:04d}_real.png")
                idx += 1

    print(f"Saved {idx} pairs to {save_dir}")


def run_evaluate(G, test_ds, output_dir: Path, ckpt_path: Path = None,
             batch_size: int = 32, num_workers: int = 2, device: str = "cuda") -> dict:
    pin_memory = (device == "cuda")
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if ckpt_path:
        G.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded checkpoint: {ckpt_path}")
    G.eval()

    noise_dim = G.noise_dim
    lpips_fn = lpips.LPIPS(net="alex").to(device)

    try:
        from flip_evaluator import flip_python_api as flip_lib
        HAS_FLIP = True
        print("FLIP library available")
    except ImportError:
        HAS_FLIP = False
        print("flip-evaluator not found, FLIP will be N/A")

    all_lpips, all_ssim, all_hausdorff, all_flip = [], [], [], []

    with torch.no_grad():
        for real_imgs, cond in tqdm(loader, desc="Evaluating", ncols=80):
            real_imgs = real_imgs.to(device)
            cond = cond.to(device)
            B = real_imgs.size(0)
            z = torch.zeros(B, noise_dim, device=device)
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
                    result = flip_lib.evaluate(r_f, f_f, "LDR")
                    all_flip.append(result[1])

    flip_mean = f"{np.mean(all_flip):.4f}" if HAS_FLIP else "N/A"

    print("\n" + "=" * 60)
    print(f"{'Method':<22} {'FLIP':>8} {'LPIPS':>8} {'SSIM':>8} {'Hausdorff':>10}")
    print("-" * 60)
    print(f"{'neural_renderer':<22} {flip_mean:>8} "
          f"{np.mean(all_lpips):>8.4f} "
          f"{np.mean(all_ssim):>8.4f} "
          f"{np.mean(all_hausdorff):>10.2f}")
    print("=" * 60)

    results = {
        "FLIP": float(np.mean(all_flip)) if HAS_FLIP else None,
        "LPIPS": float(np.mean(all_lpips)),
        "SSIM": float(np.mean(all_ssim)),
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
    device: str = "cuda",
    batch_size: int = 32,
    num_workers: int = 2,
    N: int = 4,
    param_names: list = None,
):
    if param_names is None:
        param_names = [
            "trans_x", "trans_y", "trans_z",
            "diff_r",  "diff_g",  "diff_b",
            "shininess",
            "light_x", "light_y", "light_z",
        ]

    pin_memory = (device == "cuda")
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)
    G.eval()
    output_dir = Path(output_dir)
    noise_dim = G.noise_dim

    all_real, all_fake, all_cond = [], [], []

    with torch.no_grad():
        for real_imgs, cond in loader:
            real_imgs = real_imgs.to(device)
            cond_dev = cond.to(device)
            z = torch.zeros(real_imgs.size(0), noise_dim, device=device)
            fake = G(z, cond_dev)
            all_real.append(real_imgs.cpu())
            all_fake.append(fake.cpu())
            all_cond.append(cond.cpu())
            if sum(r.size(0) for r in all_real) >= N:
                break

    real_cat = torch.cat(all_real)[:N]
    fake_cat = torch.cat(all_fake)[:N]
    cond_cat = torch.cat(all_cond)[:N]

    fig, axes = plt.subplots(N, 3, figsize=(12, N * 3.2))
    if N == 1:
        axes = axes[None, :]

    for ax, title in zip(axes[0], ["Parameters", "Reference", "Generated"]):
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

    for i in range(N):
        c = cond_cat[i].numpy()
        param_lines = []
        for j, val in enumerate(c):
            name = param_names[j] if j < len(param_names) else f"p{j}"
            param_lines.append(f"{name}: {val:.3f}")

        axes[i, 0].axis("off")
        axes[i, 0].text(
            0.05, 0.95, "\n".join(param_lines),
            transform=axes[i, 0].transAxes,
            fontsize=8.5, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#cccccc"),
        )

        axes[i, 1].imshow(to_numpy_uint8(real_cat[i]))
        axes[i, 1].axis("off")

        axes[i, 2].imshow(to_numpy_uint8(fake_cat[i]))
        axes[i, 2].axis("off")

    fig.suptitle("Neural Renderer — Parameters / Reference / Generated", fontsize=13, y=1.01)
    plt.tight_layout()
    out_path = output_dir / "results_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")
