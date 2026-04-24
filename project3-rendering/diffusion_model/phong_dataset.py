import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PhongDataset(Dataset):
    def __init__(self, data_dir: str, indices: list[int], img_size: int = 128):
        self.data_dir = Path(data_dir)
        self.indices = indices
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.indices)

    def _load_params(self, idx: int) -> torch.Tensor:
        json_path = self.data_dir / "params" / f"{idx:04d}.json"

        with open(json_path) as f:
            p = json.load(f)

        translation = [v / 20.0 for v in p["model_translation"]]        # Normalizacja do [-1, 1]
        diffuse = p["material_diffuse"]
        shininess = [(p["material_shininess"] - 3.0) / 17.0]            # Normalizacja do [0, 1]
        light = [v / 20.0 for v in p["light_position"]]                 # Normalizacja do [-1, 1]

        params = translation + diffuse + shininess + light

        return torch.tensor(params, dtype=torch.float32)

    def __getitem__(self, item):
        idx = self.indices[item]

        img_path = self.data_dir / "images" / f"image_{idx:04d}.png"

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        params = self._load_params(idx)

        return params, image
