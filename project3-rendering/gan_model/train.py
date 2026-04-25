import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt

from losses import discriminator_loss, generator_loss


def _eval_loss(G, D, loader, noise_dim, lambda_l1, device):
    G.eval(); D.eval()
    g_losses, d_losses, adv_losses, l1_losses = [], [], [], []

    with torch.no_grad():
        for real_imgs, cond in loader:
            real_imgs = real_imgs.to(device)
            cond      = cond.to(device)
            B         = real_imgs.size(0)

            z             = torch.zeros(B, noise_dim, device=device)
            fake_imgs     = G(z, cond)
            loss_G, loss_adv, loss_l1 = generator_loss(
                D(fake_imgs, cond), fake_imgs, real_imgs, lambda_l1
            )
            loss_D        = discriminator_loss(D(real_imgs, cond), D(fake_imgs, cond))

            g_losses.append(loss_G.item())
            d_losses.append(loss_D.item())
            adv_losses.append(loss_adv.item())
            l1_losses.append(loss_l1.item())

    return (float(np.mean(g_losses)), float(np.mean(d_losses)),
            float(np.mean(adv_losses)), float(np.mean(l1_losses)))


def train(
    G,
    D,
    train_ds,
    test_ds,
    output_dir:  Path,
    ckpt_dir:    Path,
    epochs:      int   = 150,
    lr:          float = 2e-4,
    betas:       tuple = (0.5, 0.999),
    lambda_l1:   float = 100.0,
    batch_size:  int   = 32,
    num_workers: int   = 2,
    save_every:  int   = 10,
    device:      str   = "cuda",
    seed:        int   = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    output_dir = Path(output_dir)
    ckpt_dir   = Path(ckpt_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    pin_memory   = (device == "cuda")
    noise_dim    = G.noise_dim

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    opt_G   = optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_D   = optim.Adam(D.parameters(), lr=lr, betas=betas)
    sched_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=epochs, eta_min=1e-5)
    sched_D = optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=epochs, eta_min=1e-5)

    preview_imgs, preview_cond = next(iter(test_loader))
    preview_real = preview_imgs[:8].to(device)
    preview_cond = preview_cond[:8].to(device)
    preview_z    = torch.zeros(8, noise_dim, device=device)

    history = {
        "train_G": [], "train_D": [],
        "val_G":   [], "val_D":   [],
    }

    best_val_g = float("inf")

    for epoch in range(1, epochs + 1):
        G.train(); D.train()
        ep_g, ep_d = [], []
        ep_adv, ep_l1 = [], []
        
        batch_bar = tqdm(train_loader,
                         desc=f"Epoch {epoch:03d}/{epochs}",
                         leave=False, unit="batch")

        for real_imgs, cond in batch_bar:
            real_imgs = real_imgs.to(device)
            cond      = cond.to(device)
            B         = real_imgs.size(0)

            z = torch.randn(B, noise_dim, device=device)
            with torch.no_grad():
                fake_imgs = G(z, cond)
            loss_D = discriminator_loss(D(real_imgs, cond), D(fake_imgs, cond))
            opt_D.zero_grad(set_to_none=True)
            loss_D.backward()
            opt_D.step()

            z = torch.randn(B, noise_dim, device=device)
            fake_imgs    = G(z, cond)
            loss_G, loss_adv, loss_l1 = generator_loss(
                D(fake_imgs, cond), fake_imgs, real_imgs, lambda_l1
            )
            opt_G.zero_grad(set_to_none=True)
            loss_G.backward()
            opt_G.step()

            ep_g.append(loss_G.item())
            ep_d.append(loss_D.item())
            ep_adv.append(loss_adv.item())
            ep_l1.append(loss_l1.item())

        sched_G.step()
        sched_D.step()

        train_g   = float(np.mean(ep_g))
        train_d   = float(np.mean(ep_d))
        train_adv = float(np.mean(ep_adv))
        train_l1  = float(np.mean(ep_l1))
        current_lr = sched_G.get_last_lr()[0]

        history["train_G"].append(train_g)
        history["train_D"].append(train_d)

        print(f"Epoch {epoch:03d}/{epochs}  "
              f"G loss={train_g:.4f} (adv={train_adv:.4f}, l1={train_l1:.4f})  "
              f"D loss={train_d:.4f}  "
              f"lr={current_lr:.2e}",
              flush=True)

        if epoch % save_every == 0 or epoch == epochs:
            val_g, val_d, val_adv, val_l1 = _eval_loss(
                G, D, test_loader, noise_dim, lambda_l1, device
            )
            history["val_G"].append(val_g)
            history["val_D"].append(val_d)

            saved = val_g < best_val_g
            if saved:
                best_val_g = val_g
                torch.save(G.state_dict(), ckpt_dir / "G_best.pth")
                torch.save(D.state_dict(), ckpt_dir / "D_best.pth")

            print(f"  └─ val  G loss={val_g:.4f} (adv={val_adv:.4f}, l1={val_l1:.4f})  "
                  f"D loss={val_d:.4f}"
                  f"  (best G={best_val_g:.4f})"
                  f"{'  [saved]' if saved else ''}",
                  flush=True)

            G.eval()
            with torch.no_grad():
                gen = G(preview_z, preview_cond)
            save_image(torch.cat([preview_real, gen]) * 0.5 + 0.5,
                       output_dir / f"sample_e{epoch:03d}.png", nrow=8)
            torch.save(G.state_dict(), ckpt_dir / f"G_{epoch:03d}.pth")
            torch.save(D.state_dict(), ckpt_dir / f"D_{epoch:03d}.pth")
            G.train()


    # ── Loss curves ───────────────────────────────────────────────────────────
    val_epochs = list(range(save_every, epochs + 1, save_every))
    if epochs not in val_epochs:
        val_epochs.append(epochs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    ax1.plot(history["train_G"], label="Train G loss")
    ax1.plot(val_epochs, history["val_G"], "o--", label="Val G loss")
    ax1.set_title("Generator loss"); ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(history["train_D"], label="Train D loss")
    ax2.plot(val_epochs, history["val_D"], "o--", label="Val D loss")
    ax2.set_title("Discriminator loss"); ax2.set_xlabel("Epoch"); ax2.legend()

    plt.suptitle("Training curves – Conditional GAN (LSGAN + L1)")
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.show()

    torch.save(G.state_dict(), ckpt_dir / "G_final.pth")
    torch.save(D.state_dict(), ckpt_dir / "D_final.pth")
    print(f"\nDone. Best val G loss: {best_val_g:.4f} → {ckpt_dir / 'G_best.pth'}")
    return history


if __name__ == "__main__":
    from dataset import get_loaders
    from models import build_models
    from config import DEVICE

    train_ds, test_ds = get_loaders(
        images_dir = Path("../data/images"),
        params_dir = Path("../data/params"),
    )
    G, D = build_models(device=DEVICE)
    train(G, D, train_ds, test_ds,
          output_dir=Path("output"), ckpt_dir=Path("checkpoints"), device=DEVICE)