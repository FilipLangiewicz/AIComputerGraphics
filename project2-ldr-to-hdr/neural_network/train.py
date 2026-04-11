import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import ExposureDataset
from model import ResUNet
from loss import ExposureLoss


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(torch.tensor(1.0 / mse)).item()


def evaluate(model, loader, device) -> float:
    model.eval()
    total_psnr = 0.0
    with torch.no_grad():
        for ldr, target in loader:
            ldr, target = ldr.to(device), target.to(device)
            pred = model(ldr)
            total_psnr += psnr(pred, target)
    return total_psnr / len(loader)


def train(
    data_root: Path,
    target: str,
    save_path:       Path,
    epochs:          int   = 100,
    batch_size:      int   = 8,
    lr:              float = 1e-4,
    alpha:           float = 0.8,
    features:        list  = [32, 64, 128, 256],
    eval_every:      int   = 5,
    num_workers:     int   = 2,
    optimizer_name:  str   = "adam",
    scheduler_name:  str   = "plateau",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- data ---
    train_dataset = ExposureDataset(root=data_root, split="train", target=target)
    test_dataset  = ExposureDataset(root=data_root, split="test",  target=target)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader   = DataLoader(test_dataset,  batch_size=1,
                               shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Train: {len(train_dataset)} samples | Test: {len(test_dataset)} samples")

    # --- model ---
    model = ResUNet(features=features).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {total_params:,}\n")

    # --- optimizer ---
    optimizers = {
        "adam":  torch.optim.Adam(model.parameters(), lr=lr),
        "adamw": torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4),
        "sgd":   torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9),
    }
    optimizer = optimizers[optimizer_name.lower()]

    # --- scheduler ---
    schedulers = {
        "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10),
        "cosine":  torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs),
        "none":    None,
    }
    scheduler = schedulers[scheduler_name.lower()]

    # --- loss ---
    criterion = ExposureLoss(alpha=alpha).to(device)

    # --- training loop ---
    best_psnr = -float("inf")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epoch_bar = tqdm(range(1, epochs + 1), desc="Training", unit="epoch")

    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0

        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{epochs}",
                         leave=False, unit="batch")

        for ldr, target in batch_bar:
            ldr, target = ldr.to(device), target.to(device)
            optimizer.zero_grad()
            pred = model(ldr)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        postfix  = {"loss": f"{avg_loss:.4f}"}

        if epoch % eval_every == 0 or epoch == epochs:
            avg_psnr = evaluate(model, test_loader, device)
            postfix["PSNR"] = f"{avg_psnr:.2f}dB"

            if scheduler and scheduler_name == "plateau":
                scheduler.step(avg_psnr)

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save({
                    "epoch":    epoch,
                    "model":    model.state_dict(),
                    "optim":    optimizer.state_dict(),
                    "psnr":     best_psnr,
                    "features": features,
                }, save_path)
                postfix["saved"] = "✓"

        elif scheduler and scheduler_name == "cosine":
            scheduler.step()

        epoch_bar.set_postfix(postfix)

    print(f"\nDone. Best PSNR: {best_psnr:.2f}dB → {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--target",    type=str,  default="under", choices=["under", "over"])
    parser.add_argument("--save_path",      type=Path, required=True)
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--batch_size",     type=int,   default=8)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--alpha",          type=float, default=0.8)
    parser.add_argument("--features",       type=int,   nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--eval_every",     type=int,   default=5)
    parser.add_argument("--num_workers",    type=int,   default=2)
    parser.add_argument("--optimizer",      type=str,   default="adam",
                        choices=["adam", "adamw", "sgd"])
    parser.add_argument("--scheduler",      type=str,   default="plateau",
                        choices=["plateau", "cosine", "none"])
    args = parser.parse_args()

    train(
        data_root = args.data_root,
        target    = args.target,
        save_path      = args.save_path,
        epochs         = args.epochs,
        batch_size     = args.batch_size,
        lr             = args.lr,
        alpha          = args.alpha,
        features       = args.features,
        eval_every     = args.eval_every,
        num_workers    = args.num_workers,
        optimizer_name = args.optimizer,
        scheduler_name = args.scheduler,
    )