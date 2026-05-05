import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, AdamW, SGD

from .model import VectorFieldNet
from .dataset import ArmadilloTeapotDataset


def chamfer_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Chamfer distance loss between two point clouds.
    pred, target: (B, N, 3)
    """
    # pairwise distances (B, N, M)
    diff = pred.unsqueeze(2) - target.unsqueeze(1)
    dist = (diff ** 2).sum(dim=-1)

    loss = dist.min(dim=2).values.mean() + dist.min(dim=1).values.mean()
    return loss


OPTIMIZERS = {"adam": Adam, "adamw": AdamW, "sgd": SGD}


def train(
    armadillo_path: str,
    teapot_path: str,
    # data
    n_points: int = 2048,
    n_samples: int = 10000,
    val_split: float = 0.1,
    augment: bool = True,
    scale_range: tuple[float, float] = (0.75, 1.25),
    # model
    local_hidden_dims: list[int] = [64, 128],
    global_hidden_dims: list[int] = [256, 512],
    output_hidden_dims: list[int] = [256, 128],
    dropout: float = 0.0,
    # training
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    optimizer: str = "adam",
    weight_decay: float = 0.0,
    # misc
    device: str = "cuda",
    checkpoint_path: str = "armadillo/checkpoints/best_model.pt",
    verbose: bool = True,
) -> VectorFieldNet:
    """
    Train VectorFieldNet to transform armadillo -> teapot.
    Returns trained model.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # dataset
    dataset = ArmadilloTeapotDataset(
        armadillo_path, teapot_path, n_points, n_samples, augment, scale_range
    )
    val_size = int(len(dataset) * val_split)
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # model
    model = VectorFieldNet(
        local_hidden_dims=local_hidden_dims,
        global_hidden_dims=global_hidden_dims,
        output_hidden_dims=output_hidden_dims,
        dropout=dropout,
    ).to(device)

    opt_cls = OPTIMIZERS.get(optimizer.lower())
    if opt_cls is None:
        raise ValueError(f"Unknown optimizer '{optimizer}'. Choose from: {list(OPTIMIZERS)}")
    opt = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            pred = model(src)
            loss = chamfer_loss(pred, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- val ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                pred = model(src)
                val_loss += chamfer_loss(pred, tgt).item()
        val_loss /= len(val_loader)

        if verbose:
            print(f"Epoch {epoch:>4}/{epochs} | train: {train_loss:.6f} | val: {val_loss:.6f}")

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

    if verbose:
        print(f"Training done. Best val loss: {best_val_loss:.6f}")

    return model


def load_model(
    checkpoint_path: str,
    local_hidden_dims: list[int] = [64, 128],
    global_hidden_dims: list[int] = [256, 512],
    output_hidden_dims: list[int] = [256, 128],
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