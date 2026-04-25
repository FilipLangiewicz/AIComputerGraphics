"""LSGAN + L1 reconstruction loss.

Discriminator: real -> 0.9 (label smoothing), fake -> 0.0
Generator:     L_G = MSE(D(fake), 1) + lambda_l1 * L1(fake, real)
"""
import torch
import torch.nn as nn

criterion_adv = nn.MSELoss()
criterion_l1  = nn.L1Loss()


def discriminator_loss(D_real: torch.Tensor, D_fake: torch.Tensor) -> torch.Tensor:
    real_lbl = torch.full_like(D_real, 0.9)
    fake_lbl = torch.zeros_like(D_fake)
    return 0.5 * (criterion_adv(D_real, real_lbl) + criterion_adv(D_fake, fake_lbl))


def generator_loss(
    D_fake:    torch.Tensor,
    fake_imgs: torch.Tensor,
    real_imgs: torch.Tensor,
    lambda_l1: float = 100.0,
):
    """Returns (loss_total, loss_adv, loss_l1)."""
    loss_adv   = criterion_adv(D_fake, torch.ones_like(D_fake))
    loss_l1    = criterion_l1(fake_imgs, real_imgs)
    loss_total = loss_adv + lambda_l1 * loss_l1
    return loss_total, loss_adv, loss_l1