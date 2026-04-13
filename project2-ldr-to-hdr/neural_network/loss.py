import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel(kernel_size: int, sigma: float, channels: int) -> torch.Tensor:
    x = torch.arange(kernel_size).float() - kernel_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel = gauss.outer(gauss)
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    return kernel


class SSIMLoss(nn.Module):
    def __init__(self, kernel_size: int = 11, sigma: float = 1.5,
                 channels: int = 3, C1: float = 0.01**2, C2: float = 0.03**2):
        super().__init__()
        self.channels     = channels
        self.C1           = C1
        self.C2           = C2
        kernel = _gaussian_kernel(kernel_size, sigma, channels)
        self.register_buffer("kernel", kernel)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        padding = self.kernel.shape[-1] // 2

        mu1 = F.conv2d(pred,   self.kernel, padding=padding, groups=self.channels)
        mu2 = F.conv2d(target, self.kernel, padding=padding, groups=self.channels)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred,     self.kernel, padding=padding, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.kernel, padding=padding, groups=self.channels) - mu2_sq
        sigma12   = F.conv2d(pred * target,   self.kernel, padding=padding, groups=self.channels) - mu1_mu2

        numerator   = (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        denominator = (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)

        ssim_map = numerator / denominator
        return 1 - ssim_map.mean()
    
class GradientLoss(nn.Module):
    def forward(self, pred, target):
        def grad(x):
            dx = x[:, :, :, 1:] - x[:, :, :, :-1]
            dy = x[:, :, 1:, :] - x[:, :, :-1, :]
            return dx, dy
        dx_p, dy_p = grad(pred)
        dx_t, dy_t = grad(target)
        return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)


class ExposureLoss(nn.Module):
    def __init__(self, alpha=0.85, beta=0.10):
        super().__init__()
        self.alpha = alpha   # L1
        self.beta  = beta    # gradient
        self.l1   = nn.L1Loss()
        self.grad = GradientLoss()

    def forward(self, pred, target):
        return self.alpha * self.l1(pred, target) \
             + self.beta  * self.grad(pred, target)