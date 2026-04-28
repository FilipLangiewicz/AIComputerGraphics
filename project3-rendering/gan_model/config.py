import torch

# Normalization constants for scene parameters
TRANS_SCALE = 20.0
LIGHT_SCALE = 20.0
SHINE_MIN   = 3.0
SHINE_MAX   = 20.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"