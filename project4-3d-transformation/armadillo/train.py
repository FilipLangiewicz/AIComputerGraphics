import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from model import VectorFieldNet
from dataset import ArmadilloTeapotDataset


def chamfer_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Chamfer distance loss. pred, target: (B, N, 3)"""
    diff = pred.unsqueeze(2) - target.unsqueeze(1)
    dist = (diff ** 2).sum(dim=-1)
    return dist.min(dim=2).values.mean() + dist.min(dim=1).values.mean()


def _build_optimizer(model, optimizer: str, lr: float, weight_decay: float):
    opts = {"adam": optim.Adam, "adamw": optim.AdamW, "sgd": optim.SGD}
    cls = opts.get(optimizer.lower())
    if cls is None:
        raise ValueError(f"Unknown optimizer '{optimizer}'. Choose from: {list(opts)}")
    return cls(model.parameters(), lr=lr, weight_decay=weight_decay)


def _build_scheduler(optimizer, scheduler: str, epochs: int):
    s = scheduler.lower() if scheduler else "none"
    if s == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif s == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.5)
    elif s == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    elif s == "none" or s is None:
        return None
    else:
        raise ValueError(f"Unknown scheduler '{scheduler}'. Choose from: cosine, step, plateau, none")


def _eval(model, loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            losses.append(chamfer_loss(model(src), tgt).item())
    return float(np.mean(losses))


def train(
    armadillo_path: str,
    teapot_path: str,
    # data
    n_points: int = 2048,
    n_samples: int = 10000,
    val_split: float = 0.1,
    augment: bool = True,
    scale_range: tuple = (0.75, 1.25),
    # model
    local_hidden_dims: list = [64, 128],
    global_hidden_dims: list = [256, 512],
    output_hidden_dims: list = [256, 128],
    dropout: float = 0.0,
    # training
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    optimizer: str = "adam",
    weight_decay: float = 0.0,
    scheduler: str = "cosine",   # "cosine" | "step" | "plateau" | "none"
    save_every: int = 10,
    # misc
    device: str = "cuda",
    ckpt_dir: str = "armadillo/checkpoints",
    seed: int = 42,
    num_workers: int = 2,
) -> VectorFieldNet:
    """Train VectorFieldNet to transform armadillo -> teapot. Returns trained model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # dataset
    dataset = ArmadilloTeapotDataset(
        armadillo_path, teapot_path, n_points, n_samples, augment, scale_range
    )
    val_size = int(len(dataset) * val_split)
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    # model
    model = VectorFieldNet(
        local_hidden_dims=local_hidden_dims,
        global_hidden_dims=global_hidden_dims,
        output_hidden_dims=output_hidden_dims,
        dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"VectorFieldNet  params: {n_params:,}")

    opt   = _build_optimizer(model, optimizer, lr, weight_decay)
    sched = _build_scheduler(opt, scheduler, epochs)

    history = {"train": [], "val": [], "val_epochs": []}
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        ep_losses = []

        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{epochs}",
                         leave=False, unit="batch")
        for src, tgt in batch_bar:
            src, tgt = src.to(device), tgt.to(device)
            loss = chamfer_loss(model(src), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_losses.append(loss.item())
            batch_bar.set_postfix(loss=f"{loss.item():.6f}")

        train_loss = float(np.mean(ep_losses))
        history["train"].append(train_loss)

        current_lr = opt.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d}/{epochs}  loss={train_loss:.6f}  lr={current_lr:.2e}")

        # scheduler step (plateau needs metric)
        if sched is not None:
            if isinstance(sched, optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(train_loss)
            else:
                sched.step()

        # validation
        if epoch % save_every == 0 or epoch == epochs:
            val_loss = _eval(model, val_loader, device)
            history["val"].append(val_loss)
            history["val_epochs"].append(epoch)

            saved = val_loss < best_val
            if saved:
                best_val = val_loss
                torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

            print(f"  └─ val  loss={val_loss:.6f}  "
                  f"(best={best_val:.6f})"
                  f"{'  [saved]' if saved else ''}")

            torch.save(model.state_dict(), ckpt_dir / f"model_e{epoch:03d}.pt")

    # loss curves
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(history["train"]) + 1), history["train"], label="Train loss")
    ax.plot(history["val_epochs"], history["val"], "o--", label="Val loss")
    ax.set_title("Chamfer loss — Armadillo → Teapot")
    ax.set_xlabel("Epoch")
    ax.legend()
    plt.tight_layout()
    plt.savefig(ckpt_dir / "loss_curve.png", dpi=150)
    plt.show()

    torch.save(model.state_dict(), ckpt_dir / "final_model.pt")
    print(f"\nDone. Best val loss: {best_val:.6f} → {ckpt_dir / 'best_model.pt'}")
    return model


def load_model(
    checkpoint_path: str,
    local_hidden_dims: list = [64, 128],
    global_hidden_dims: list = [256, 512],
    output_hidden_dims: list = [256, 128],
    dropout: float = 0.0,
    device: str = "cuda",
) -> VectorFieldNet:
    """Load a trained VectorFieldNet from checkpoint."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = VectorFieldNet(
        local_hidden_dims=local_hidden_dims,
        global_hidden_dims=global_hidden_dims,
        output_hidden_dims=output_hidden_dims,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model