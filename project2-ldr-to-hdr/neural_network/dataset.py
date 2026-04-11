from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

def _load(path: Path) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)

class ExposureDataset(Dataset):
    def __init__(self, root: Path, split: str = "train", target: str = "under"):
        assert target in ("under", "over"), "target must be 'under' or 'over'"
        root = Path(root)
        self.ldr_dir    = root / split / "ldr"
        self.target_dir = root / split / target
        self.files = sorted(
            f.name for f in self.ldr_dir.iterdir()
            if f.suffix.lower() in IMG_EXTENSIONS
        )
        if not self.files:
            raise RuntimeError(f"No images found in {self.ldr_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        fname = self.files[idx]
        return _load(self.ldr_dir / fname), _load(self.target_dir / fname)