import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from model import MotionDenoiser
from diffusion import GaussianDiffusion
from dataset import MotionDataset


def _build_optimizer(model: torch.nn.Module, name: str, lr: float, weight_decay: float):
    opts = {"adam": optim.Adam, "adamw": optim.AdamW}
    cls = opts.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown optimizer '{name}'. Choose from: {list(opts)}")
    return cls(model.parameters(), lr=lr, weight_decay=weight_decay)


def _build_scheduler(optimizer, name: str, epochs: int):
    s = (name or "none").lower()
    if s == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif s == "none":
        return None
    raise ValueError(f"Unknown scheduler '{name}'. Choose from: cosine, none")


def _sample_qualitative(model: MotionDenoiser, diffusion: GaussianDiffusion,
                         device: torch.device, n_samples: int,
                         num_classes: int, guidance_scale: float,
                         out_path: Path, norm_stats: np.ndarray = None) -> None:
    model.eval()
    samples = {}
    for cls_id in range(num_classes):
        labels = torch.full((n_samples,), cls_id, dtype=torch.long, device=device)
        motion = diffusion.sample(model, labels, guidance_scale=guidance_scale).cpu()
        if norm_stats is not None:                       
            mean, std = norm_stats[0], norm_stats[1]
            motion = motion * std + mean
        samples[cls_id] = motion
    torch.save(samples, out_path)
    model.train()


def train(
    dataset: MotionDataset,
    norm_stats_path: str = None,
    # model
    d_model: int = 256,
    nhead: int = 4,
    num_layers: int = 4,
    dropout: float = 0.1,
    # diffusion
    timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    cfg_drop_prob: float = 0.1,
    guidance_scale: float = 3.0,
    # training
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-4,
    optimizer: str = "adamw",
    weight_decay: float = 1e-4,
    scheduler: str = "cosine",
    grad_clip: float = 1.0,
    vel_loss_weight: float = 0.1,
    save_every: int = 50,
    # qualitative eval
    eval_every: int = 25,
    eval_samples: int = 4,
    # misc
    device: str = "cuda",
    ckpt_dir: str = "checkpoints",
    resume_from: str = None,
    resume_optimizer: bool = True,
    resume_scheduler: bool = True,
    seed: int = 42,
    num_workers: int = 4,
) -> MotionDenoiser:
    """Train MotionDenoiser. Returns trained model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    norm_stats = np.load(norm_stats_path) if norm_stats_path and Path(norm_stats_path).exists() else None
    ckpt_dir = Path(ckpt_dir)
    (ckpt_dir / "samples").mkdir(parents=True, exist_ok=True)

    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=num_workers, pin_memory=device.type == "cuda")

    n_frames    = dataset.sequences.shape[1]
    n_joints    = dataset.sequences.shape[2]
    num_classes = int(dataset.labels.max().item()) + 1

    model = MotionDenoiser(
        n_joints=n_joints, n_frames=n_frames, d_model=d_model,
        nhead=nhead, num_layers=num_layers, num_classes=num_classes, dropout=dropout,
    ).to(device)

    diffusion = GaussianDiffusion(timesteps=timesteps, beta_start=beta_start,
                                  beta_end=beta_end).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MotionDenoiser  params: {n_params:,}  |  device: {device}")
    print(f"dataset: {len(dataset)}  classes: {num_classes}")

    opt   = _build_optimizer(model, optimizer, lr, weight_decay)
    sched = _build_scheduler(opt, scheduler, epochs)

    start_epoch = 1
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        if resume_optimizer and "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if resume_scheduler and sched is not None and ckpt.get("scheduler") is not None:
            sched.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"resumed from: {resume_from}  (epoch {start_epoch})")

    history = []

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        ep_losses = []

        bar = tqdm(loader, desc=f"Epoch {epoch:03d}/{epochs}", leave=False, unit="batch")
        for sequences_b, labels_b in bar:
            sequences_b = sequences_b.to(device)
            labels_b    = labels_b.to(device)
            t = torch.randint(0, diffusion.T, (sequences_b.shape[0],), device=device)

            loss = diffusion.p_losses(model, sequences_b, t, labels_b,
                          cfg_drop_prob, vel_loss_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            ep_losses.append(loss.item())
            bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = float(np.mean(ep_losses))
        history.append(train_loss)
        current_lr = opt.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d}/{epochs}  loss={train_loss:.4f}  lr={current_lr:.2e}", end="")

        if sched is not None:
            sched.step()

        if epoch % eval_every == 0 or epoch == epochs:
            out_path = ckpt_dir / "samples" / f"samples_e{epoch:03d}.pt"
            _sample_qualitative(model, diffusion, device, eval_samples,
                                 num_classes, guidance_scale, out_path, norm_stats)
            print(f"  |  samples → {out_path.name}", end="")

        print()

        if epoch % save_every == 0 or epoch == epochs:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict() if sched else None,
            }, ckpt_dir / f"ckpt_e{epoch:03d}.pt")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(start_epoch, start_epoch + len(history)), history)
    ax.set_title("Train loss (MSE noise)")
    ax.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(ckpt_dir / "loss_curve.png", dpi=150)
    plt.close()

    torch.save({"model": model.state_dict(), "epoch": epochs}, ckpt_dir / "final_model.pt")
    print(f"\nDone → {ckpt_dir / 'final_model.pt'}")
    return model


def load_model(
    checkpoint_path: str,
    n_joints: int = 15,
    n_frames: int = 48,
    d_model: int = 256,
    nhead: int = 4,
    num_layers: int = 4,
    num_classes: int = 2,
    dropout: float = 0.1,
    device: str = "cuda",
) -> MotionDenoiser:
    """Load a trained MotionDenoiser from checkpoint."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = MotionDenoiser(
        n_joints=n_joints, n_frames=n_frames, d_model=d_model,
        nhead=nhead, num_layers=num_layers, num_classes=num_classes, dropout=dropout,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    return model