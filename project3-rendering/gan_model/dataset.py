import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

from config import TRANS_SCALE, LIGHT_SCALE, SHINE_MIN, SHINE_MAX


def normalize_params(
    p: dict,
    trans_scale: float = TRANS_SCALE,
    light_scale: float = LIGHT_SCALE,
    shine_min:   float = SHINE_MIN,
    shine_max:   float = SHINE_MAX,
) -> np.ndarray:
    """Normalize scene parameters to [-1, 1].

    Stores relative light position (light_pos - model_translation)
    instead of absolute coordinates.
    """
    mt = np.array(p["model_translation"])
    lp = np.array(p["light_position"])

    tx, ty, tz            = mt / trans_scale
    r,  g,  b             = np.array(p["material_diffuse"]) * 2.0 - 1.0
    shine                 = ((p["material_shininess"] - shine_min)
                             / (shine_max - shine_min)) * 2.0 - 1.0
    rel_lx, rel_ly, rel_lz = (lp - mt) / (2.0 * light_scale)

    return np.array([tx, ty, tz, r, g, b, shine,
                     rel_lx, rel_ly, rel_lz], dtype=np.float32)


class PhongDataset(Dataset):
    def __init__(
        self,
        indices,
        images_dir: Path,
        params_dir: Path,
        img_size:   int = 128,
    ):
        self.indices    = list(indices)
        self.images_dir = Path(images_dir)
        self.params_dir = Path(params_dir)
        self.transform  = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        img_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            p = self.images_dir / f"image_{i:04d}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            raise FileNotFoundError(f"Missing image: image_{i:04d}.[png/jpg]")

        img_tensor = self.transform(Image.open(img_path).convert("RGB"))

        with open(self.params_dir / f"{i:04d}.json") as f:
            params = json.load(f)

        cond = torch.from_numpy(normalize_params(params))
        return img_tensor, cond


def get_datasets(
    images_dir: Path,
    params_dir: Path,
    train_end:  int = 2400,
    test_start: int = 2400,
    total:      int = 3000,
    img_size:   int = 128,
):
    train_ds = PhongDataset(range(0, train_end),     images_dir, params_dir, img_size)
    test_ds  = PhongDataset(range(test_start, total), images_dir, params_dir, img_size)
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")
    return train_ds, test_ds