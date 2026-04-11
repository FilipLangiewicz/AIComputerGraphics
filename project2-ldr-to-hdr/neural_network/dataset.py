import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

TEST_SCENES     = {f"C{i:02d}" for i in range(40, 47)}
IMG_EXTENSIONS  = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
PATCH_SIZE      = 256
PATCHES_PER_IMG = 50


def _find_file(directory: Path, scene: str) -> Path | None:
    for f in directory.iterdir():
        if f.stem.upper().startswith(scene.upper()) and f.suffix.lower() in IMG_EXTENSIONS:
            return f
    return None


def _load_image(path: Path) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


def _augment(ldr: torch.Tensor, target: torch.Tensor, patch_size: int):
    _, h, w = ldr.shape
    i = torch.randint(0, h - patch_size + 1, (1,)).item()
    j = torch.randint(0, w - patch_size + 1, (1,)).item()
    ldr    = TF.crop(ldr,    i, j, patch_size, patch_size)
    target = TF.crop(target, i, j, patch_size, patch_size)

    if random.random() > 0.5:
        ldr, target = TF.hflip(ldr), TF.hflip(target)
    if random.random() > 0.5:
        ldr, target = TF.vflip(ldr), TF.vflip(target)

    angle = random.choice([0, 90, 180, 270])
    if angle != 0:
        ldr    = TF.rotate(ldr,    angle)
        target = TF.rotate(target, angle)

    return ldr, target


class ExposureDataset(Dataset):
    def __init__(self, ldr_dir: Path, target_dir: Path,
                 train: bool = True,
                 patch_size: int = PATCH_SIZE,
                 patches_per_img: int = PATCHES_PER_IMG):
        self.ldr_dir         = Path(ldr_dir)
        self.target_dir      = Path(target_dir)
        self.train           = train
        self.patch_size      = patch_size
        self.patches_per_img = patches_per_img
        self.pairs           = self._build_pairs()

        if not self.pairs:
            raise RuntimeError(f"No pairs found in {ldr_dir} / {target_dir}")

    def _build_pairs(self) -> list[tuple[Path, Path]]:
        pairs = []
        for ldr_file in sorted(self.ldr_dir.iterdir()):
            if ldr_file.suffix.lower() not in IMG_EXTENSIONS:
                continue
            scene   = ldr_file.name.split("_")[0]
            is_test = scene in TEST_SCENES
            if self.train == is_test:
                continue
            target = _find_file(self.target_dir, scene)
            if target is None:
                continue
            pairs.append((ldr_file, target))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_img if self.train else len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ldr_path, target_path = self.pairs[idx % len(self.pairs)]
        ldr    = _load_image(ldr_path)
        target = _load_image(target_path)
        if self.train:
            ldr, target = _augment(ldr, target, self.patch_size)
        return ldr, target