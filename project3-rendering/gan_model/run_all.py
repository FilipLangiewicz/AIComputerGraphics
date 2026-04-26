from pathlib import Path
from config import DEVICE
from dataset import get_loaders
from models import build_models
from train import train
from evaluate import evaluate, visualize_results

# ── Paths ─────────────────────────────────────────────────────────────────────
IMAGES_DIR = Path("../data/images")
PARAMS_DIR = Path("../data/params")
OUTPUT_DIR = Path("output")
CKPT_DIR   = Path("checkpoints")

# ── Data ──────────────────────────────────────────────────────────────────────
train_loader, test_loader = get_loaders(
    images_dir  = IMAGES_DIR,
    params_dir  = PARAMS_DIR,
    batch_size  = 32,
)

# ── Models ────────────────────────────────────────────────────────────────────
G, D = build_models(
    noise_dim  = 64,
    cond_dim   = 10,
    features_g = 64,
    features_d = 64,
    device     = DEVICE,
)

# ── Train ─────────────────────────────────────────────────────────────────────
history = train(
    G, D,
    train_loader = train_loader,
    test_loader  = test_loader,
    output_dir   = OUTPUT_DIR,
    ckpt_dir     = CKPT_DIR,
    epochs       = 150,
    lr           = 2e-4,
    lambda_l1    = 100.0,
    save_every   = 10,
    device       = DEVICE,
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
evaluate(
    G,
    test_loader = test_loader,
    output_dir  = OUTPUT_DIR,
    ckpt_path   = CKPT_DIR / "G_final.pth",
    device      = DEVICE,
)

visualize_results(
    G,
    test_loader = test_loader,
    output_dir  = OUTPUT_DIR,
    device      = DEVICE,
)