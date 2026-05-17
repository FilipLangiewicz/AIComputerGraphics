import numpy as np
import torch
from torch.utils.data import Dataset


class MotionDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.sequences = torch.tensor(data["sequences"], dtype=torch.float32)
        self.labels = torch.tensor(data["labels"], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx]