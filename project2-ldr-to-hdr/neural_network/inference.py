from pathlib import Path

import cv2
import numpy as np
import torch

from model import ResUNet

EXTENSIONS  = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
TEST_SCENES = {f"C{i:02d}" for i in range(40, 47)}


def load_model(checkpoint_path: Path, device: torch.device) -> ResUNet:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = ResUNet(features=ckpt["features"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {checkpoint_path.name}  (epoch {ckpt['epoch']}, PSNR {ckpt['psnr']:.2f}dB)")
    return model


def infer_single(
    image_path: Path,
    model: ResUNet,
    device: torch.device,
) -> np.ndarray:
    """Run inference on a single image. Returns uint8 RGB numpy array."""
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32) / 255.0

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(tensor)

    out = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (out * 255).clip(0, 255).astype(np.uint8)


def run_inference(
    ldr_dir:           Path,
    under_model_path:  Path,
    over_model_path:   Path,
    out_dir:           Path,
    test_only:         bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model_under = load_model(under_model_path, device)
    model_over  = load_model(over_model_path,  device)

    out_under = Path(out_dir) / "under"
    out_over  = Path(out_dir) / "over"
    out_under.mkdir(parents=True, exist_ok=True)
    out_over.mkdir(parents=True, exist_ok=True)

    ldr_files = sorted(f for f in Path(ldr_dir).iterdir() if f.suffix.lower() in EXTENSIONS)

    if test_only:
        ldr_files = [f for f in ldr_files if f.name.split("_")[0].upper() in TEST_SCENES]
        print(f"test_only=True — processing {len(ldr_files)} test scenes (C40-C46)\n")
    else:
        print(f"test_only=False — processing all {len(ldr_files)} scenes\n")

    for ldr_path in ldr_files:
        result_under = infer_single(ldr_path, model_under, device)
        result_over  = infer_single(ldr_path, model_over,  device)

        dst_name = ldr_path.stem + ".png"
        cv2.imwrite(str(out_under / dst_name), cv2.cvtColor(result_under, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_over  / dst_name), cv2.cvtColor(result_over,  cv2.COLOR_RGB2BGR))

        print(f"  {ldr_path.name}")

    print(f"\nDone. Results saved to {out_dir}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ldr_dir",          type=Path, required=True)
    ap.add_argument("--under_model_path", type=Path, required=True)
    ap.add_argument("--over_model_path",  type=Path, required=True)
    ap.add_argument("--out_dir",          type=Path, required=True)
    ap.add_argument("--test_only",        action="store_true", default=True)
    ap.add_argument("--all",              dest="test_only", action="store_false",
                    help="Process all scenes, not just test (C40-C46)")
    args = ap.parse_args()

    run_inference(
        ldr_dir          = args.ldr_dir,
        under_model_path = args.under_model_path,
        over_model_path  = args.over_model_path,
        out_dir          = args.out_dir,
        test_only        = args.test_only,
    )