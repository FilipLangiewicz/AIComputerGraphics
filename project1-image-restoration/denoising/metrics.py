import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
import lpips

# LPIPS network loaded once at module level (expensive to initialize)
_lpips_net = None

def _get_lpips():
    global _lpips_net
    if _lpips_net is None:
        _lpips_net = lpips.LPIPS(net="vgg")
    return _lpips_net


def sne(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Squared Norm Error — lower is better."""
    return torch.sum((pred - target) ** 2).item()


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio — higher is better."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(max_val**2 / mse).item()


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Structural Similarity Index — higher is better."""
    return ssim_fn(pred, target, data_range=1.0).item()


def lpips_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Learned Perceptual Image Patch Similarity — lower is better."""
    net = _get_lpips()
    net.to(pred.device)
    net.eval()
    # lpips expects images in [-1, 1]
    pred_n   = pred   * 2 - 1
    target_n = target * 2 - 1
    with torch.no_grad():
        return net(pred_n, target_n).mean().item()


def compute_all(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute all four metrics at once. Tensors shape: (B, C, H, W), range [0, 1]."""
    return {
        "SNE":   sne(pred, target),
        "PSNR":  psnr(pred, target),
        "SSIM":  ssim(pred, target),
        "LPIPS": lpips_score(pred, target),
    }
