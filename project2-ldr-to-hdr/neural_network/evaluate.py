from pathlib import Path

import cv2
import numpy as np
import torch
import lpips

EXTENSIONS  = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
TEST_SCENES = {f"C{i:02d}" for i in range(40, 47)}


def scene_id(filename: str) -> str:
    stem = Path(filename).stem
    for sep in ("-", "+", "_"):
        if sep in stem:
            return stem.split(sep)[0].upper()
    return stem.upper()


def index_dir(d: Path) -> dict[str, Path]:
    return {
        scene_id(p.name): p
        for p in sorted(d.iterdir())
        if p.suffix.lower() in EXTENSIONS
    }


def load_rgb_uint8(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    return img.astype(np.uint8)


def to_tensor_lpips(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert uint8 HxWx3 to lpips-ready tensor (1x3xHxW) in [-1, 1]."""
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0
    return t.unsqueeze(0).to(device)


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred.astype(np.float32) - target.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def evaluate(
    gt_under_dir:   Path,
    gt_over_dir:    Path,
    pred_under_dir: Path,
    pred_over_dir:  Path,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net="alex").to(device)
    loss_fn.eval()

    gt_under   = index_dir(Path(gt_under_dir))
    gt_over    = index_dir(Path(gt_over_dir))
    pred_under = index_dir(Path(pred_under_dir))
    pred_over  = index_dir(Path(pred_over_dir))

    scenes = sorted(
        set(gt_under) & set(gt_over) & set(pred_under) & set(pred_over) & TEST_SCENES
    )

    if not scenes:
        raise RuntimeError("No matching test scenes found. Check your directories.")

    results = {}

    for scene in scenes:
        gt_u  = load_rgb_uint8(gt_under[scene])
        gt_o  = load_rgb_uint8(gt_over[scene])
        pr_u  = load_rgb_uint8(pred_under[scene])
        pr_o  = load_rgb_uint8(pred_over[scene])

        # resize GT to match predicted (predicted = LDR resolution)
        h, w = pr_u.shape[:2]
        gt_u = cv2.resize(gt_u, (w, h), interpolation=cv2.INTER_CUBIC)
        gt_o = cv2.resize(gt_o, (w, h), interpolation=cv2.INTER_CUBIC)

        psnr_u = psnr(pr_u, gt_u)
        psnr_o = psnr(pr_o, gt_o)

        with torch.no_grad():
            lpips_u = loss_fn(to_tensor_lpips(pr_u, device),
                              to_tensor_lpips(gt_u, device)).item()
            lpips_o = loss_fn(to_tensor_lpips(pr_o, device),
                              to_tensor_lpips(gt_o, device)).item()

        results[scene] = {
            "psnr_under":  psnr_u,
            "psnr_over":   psnr_o,
            "lpips_under": lpips_u,
            "lpips_over":  lpips_o,
        }

        print(f"  {scene}  PSNR under={psnr_u:.2f}dB  over={psnr_o:.2f}dB  "
              f"LPIPS under={lpips_u:.4f}  over={lpips_o:.4f}")

    psnr_u_mean  = np.mean([r["psnr_under"]  for r in results.values()])
    psnr_o_mean  = np.mean([r["psnr_over"]   for r in results.values()])
    lpips_u_mean = np.mean([r["lpips_under"] for r in results.values()])
    lpips_o_mean = np.mean([r["lpips_over"]  for r in results.values()])

    print(f"\n{'='*60}")
    print(f"{'Metric':<20} {'PSNR':>10} {'LPIPS':>10}")
    print(f"{'-'*60}")
    print(f"{'underexposed':<20} {psnr_u_mean:>10.2f} {lpips_u_mean:>10.4f}")
    print(f"{'overexposed':<20} {psnr_o_mean:>10.2f} {lpips_o_mean:>10.4f}")
    print(f"{'='*60}\n")

    return results