import argparse
import random
from pathlib import Path

import cv2
import numpy as np

EXTENSIONS  = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
TEST_SCENES = {f"C{i:02d}" for i in range(40, 47)}

BOLD  = "\033[1m";  RESET  = "\033[0m"
GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"; CYAN = "\033[96m"


# ── helpers ────────────────────────────────────────────────────────────────

def scene_id(stem: str) -> str:
    """Return the scene prefix (e.g. 'C01') from any filename stem.

    Handles: C01_LDR.tif  ->  C01
             C01-27.jpg   ->  C01
             C01+27.jpg   ->  C01
    """
    for sep in ("-", "+", "_"):
        if sep in stem:
            return stem.split(sep)[0].upper()
    return stem.upper()


def index_dir(d: Path) -> dict[str, Path]:
    return {
        scene_id(p.stem): p
        for p in sorted(d.iterdir())
        if p.suffix.lower() in EXTENSIONS
    }


def build_triplets(ldr_dir: Path, under_dir: Path, over_dir: Path) -> list[dict]:
    ldr   = index_dir(ldr_dir)
    under = index_dir(under_dir)
    over  = index_dir(over_dir)

    complete = sorted(set(ldr) & set(under) & set(over))

    skipped_ldr   = set(ldr)   - set(under) - set(over)
    skipped_under = set(under) - set(ldr)
    skipped_over  = set(over)  - set(ldr)

    if not complete:
        raise FileNotFoundError(
            f"No complete triplets found.\n"
            f"  LDR scenes   : {sorted(ldr)}\n"
            f"  Under scenes : {sorted(under)}\n"
            f"  Over scenes  : {sorted(over)}"
        )

    print(f"\n{BOLD}Triplet matching{RESET}")
    print(f"  {GREEN}Complete triplets : {len(complete)}{RESET}")
    if skipped_ldr:
        print(f"  {YELLOW}LDR-only (skipped): {sorted(skipped_ldr)}{RESET}")
    if skipped_under:
        print(f"  {YELLOW}Under-only (skipped): {sorted(skipped_under)}{RESET}")
    if skipped_over:
        print(f"  {YELLOW}Over-only (skipped): {sorted(skipped_over)}{RESET}")
    print()

    return [
        {"scene": s, "ldr": ldr[s], "under": under[s], "over": over[s]}
        for s in complete
    ]


def load_rgb(path: Path) -> np.ndarray:
    """Load image as uint8 RGB regardless of source bit depth."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Cannot read: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    return img.astype(np.uint8)


def align_sizes(*imgs: np.ndarray) -> list[np.ndarray]:
    """Resize all images to the shape of the first one."""
    h, w = imgs[0].shape[:2]
    return [
        cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        if img.shape[:2] != (h, w) else img
        for img in imgs
    ]


def ensure_min_size(*imgs: np.ndarray, size: int) -> list[np.ndarray]:
    """Upscale all images if the smallest dimension is below `size`."""
    h, w = imgs[0].shape[:2]
    if min(h, w) < size:
        scale = size / min(h, w) + 0.01
        new_w, new_h = int(w * scale), int(h * scale)
        imgs = tuple(
            cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            for img in imgs
        )
    return list(imgs)


def random_crop(*imgs: np.ndarray, size: int) -> list[np.ndarray]:
    """Random crop applied identically to all images in the triplet."""
    h, w = imgs[0].shape[:2]
    y = random.randint(0, h - size)
    x = random.randint(0, w - size)
    return [img[y:y + size, x:x + size] for img in imgs]


def augment(*imgs: np.ndarray) -> list[np.ndarray]:
    """Identical random flip + rotation for all images in the triplet."""
    imgs = list(imgs)
    if random.random() > 0.5:
        imgs = [np.fliplr(img) for img in imgs]
    if random.random() > 0.5:
        imgs = [np.flipud(img) for img in imgs]
    k = random.choice([0, 1, 2, 3])
    if k:
        imgs = [np.ascontiguousarray(np.rot90(img, k)) for img in imgs]
    return imgs


def save_png(img: np.ndarray, path: Path):
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ── main ────────────────────────────────────────────────────────────────────

def build_dataset(
    ldr_dir:     Path,
    under_dir:   Path,
    over_dir:    Path,
    out_dir:     Path,
    patch_size:  int = 256,
    patches_per: int = 10,
    seed:        int = 42,
):
    random.seed(seed)
    np.random.seed(seed)

    triplets = build_triplets(Path(ldr_dir), Path(under_dir), Path(over_dir))

    for split in ("train", "test"):
        for sub in ("ldr", "under", "over"):
            (out_dir / split / sub).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "test": 0}

    for idx, triplet in enumerate(triplets):
        scene   = triplet["scene"]
        is_test = scene in TEST_SCENES
        split   = "test" if is_test else "train"
        tag     = f"{CYAN}TEST {RESET}" if is_test else f"{GREEN}train{RESET}"

        print(f"[{idx + 1:02d}/{len(triplets)}] {BOLD}{scene}{RESET} -> {tag}  "
              f"({triplet['ldr'].name})")

        ldr_img   = load_rgb(triplet["ldr"])
        under_img = load_rgb(triplet["under"])
        over_img  = load_rgb(triplet["over"])

        ldr_img, under_img, over_img = align_sizes(ldr_img, under_img, over_img)
        ldr_img, under_img, over_img = ensure_min_size(
            ldr_img, under_img, over_img, size=patch_size
        )

        for p in range(patches_per):
            crops = random_crop(ldr_img, under_img, over_img, size=patch_size)
            if not is_test:
                crops = augment(*crops)
            else:
                crops = [np.ascontiguousarray(c) for c in crops]

            lp, up, op = crops
            fname = f"{scene}_{p:04d}.png"
            save_png(lp, out_dir / split / "ldr"   / fname)
            save_png(up, out_dir / split / "under"  / fname)
            save_png(op, out_dir / split / "over"   / fname)
            counts[split] += 1

    print(f"\n{BOLD}Done{RESET}")
    print(f"  {GREEN}Train patches : {counts['train']}{RESET}")
    print(f"  {CYAN}Test  patches : {counts['test']}{RESET}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build patch dataset for exposure synthesis (EV +/-2.7).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--ldr_dir",     type=Path, required=True,
                    help="Directory with selected LDR images (output of prepare_data.py)")
    ap.add_argument("--under_dir",   type=Path, required=True,
                    help="Directory with EV=-2.7 images (Bracketed_images-27)")
    ap.add_argument("--over_dir",    type=Path, required=True,
                    help="Directory with EV=+2.7 images (Bracketed_images+27)")
    ap.add_argument("--out_dir",     type=Path, required=True,
                    help="Output root directory for the patch dataset")
    ap.add_argument("--patch_size",  type=int,  default=256)
    ap.add_argument("--patches_per", type=int,  default=10,
                    help="Number of patches per scene image")
    ap.add_argument("--seed",        type=int,  default=42)
    args = ap.parse_args()

    build_dataset(
        ldr_dir     = args.ldr_dir,
        under_dir   = args.under_dir,
        over_dir    = args.over_dir,
        out_dir     = args.out_dir,
        patch_size  = args.patch_size,
        patches_per = args.patches_per,
        seed        = args.seed,
    )