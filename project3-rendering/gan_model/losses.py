"""LSGAN + L1 reconstruction loss.

Discriminator: real -> 0.9 (label smoothing), fake -> 0.0
Generator:     L_G = MSE(D(fake), 1) + lambda_l1 * L1(fake, real)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


criterion_adv = nn.MSELoss()

def gaussian_blur_mask(mask: torch.Tensor, sigma: float = 3.0) -> torch.Tensor:
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    x = torch.arange(kernel_size, dtype=torch.float32, device=mask.device)
    x -= kernel_size // 2
    gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
    gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
    gauss_2d /= gauss_2d.sum()
    
    kernel = gauss_2d.expand(1, 1, kernel_size, kernel_size)
    padding = kernel_size // 2
    
    B, C, H, W = mask.shape
    mask_flat = mask.contiguous().view(B * C, 1, H, W)
    soft = F.conv2d(mask_flat, kernel, padding=padding)
    return soft.view(B, C, H, W).clamp(0, 1)


def masked_l1(
    fake: torch.Tensor,
    real: torch.Tensor,
    fg_weight: float = 50.0,
    threshold: float = 0.05,
    sigma: float = 3.0,          
) -> torch.Tensor:
    real_01 = (real + 1.0) / 2.0
    hard_mask = (real_01.abs().mean(dim=1, keepdim=True) > threshold).float()
    hard_mask = hard_mask.expand_as(real)
    
    soft_mask = gaussian_blur_mask(hard_mask, sigma=sigma)
    
    weights = 1.0 + (fg_weight - 1.0) * soft_mask  
    return (weights * (fake - real).abs()).mean()


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
    loss_l1    = masked_l1(fake_imgs, real_imgs)
    loss_total = loss_adv + lambda_l1 * loss_l1
    return loss_total, loss_adv, loss_l1