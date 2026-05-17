import torch
import torch.nn as nn
import numpy as np


class GaussianDiffusion:
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = timesteps

        steps = torch.arange(timesteps + 1) / timesteps
        f = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2) ** 2
        alphas_cumprod_cos = f / f[0]
        betas = torch.clamp(1 - alphas_cumprod_cos[1:] / alphas_cumprod_cos[:-1], max=0.999)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register = lambda name, val: setattr(self, name, val)
        self.register("betas", betas)
        self.register("alphas_cumprod", alphas_cumprod)
        self.register("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register("sqrt_one_minus_alphas_cumprod", (1 - alphas_cumprod).sqrt())
        self.register("posterior_variance", betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod))

    def to(self, device):
        for attr in ["betas", "alphas_cumprod", "sqrt_alphas_cumprod",
                     "sqrt_one_minus_alphas_cumprod", "posterior_variance"]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """Forward process: add noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_a = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_1a = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        return sqrt_a * x0 + sqrt_1a * noise, noise

    def p_losses(self, model, x0, t, class_label, cfg_drop_prob=0.1):
        noise = torch.randn_like(x0)
        x_noisy, _ = self.q_sample(x0, t, noise)

        mask = torch.rand(class_label.shape[0], device=class_label.device) < cfg_drop_prob
        label_input = class_label.clone()
        label_input[mask] = model.null_class_idx

        noise_pred = model(x_noisy, t, label_input)
        loss_noise = nn.functional.mse_loss(noise_pred, noise)

        # recover approx x0 from noise prediction
        sqrt_a  = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_1a = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        x0_pred = (x_noisy - sqrt_1a * noise_pred) / sqrt_a.clamp(min=1e-5)

        # velocity loss on motion space, not noise space
        pred_vel = x0_pred[:, 1:] - x0_pred[:, :-1]
        true_vel = x0[:, 1:]     - x0[:, :-1]
        loss_vel = nn.functional.mse_loss(pred_vel, true_vel)

        return loss_noise + 0.1 * loss_vel

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: int,
                 class_label: torch.Tensor, guidance_scale: float = 3.0) -> torch.Tensor:
        """Single reverse step with classifier-free guidance."""
        t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)

        null_label = torch.full_like(class_label, model.null_class_idx)
        eps_cond = model(x, t_tensor, class_label)
        eps_uncond = model(x, t_tensor, null_label)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        beta_t = self.betas[t]
        sqrt_1a = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_a = (1.0 / (1.0 - beta_t).sqrt())

        mean = sqrt_recip_a * (x - beta_t / sqrt_1a * eps)

        if t > 0:
            noise = torch.randn_like(x)
            return mean + self.posterior_variance[t].sqrt() * noise
        return mean

    @torch.no_grad()
    def sample(self, model: nn.Module, class_label: torch.Tensor,
               n_frames: int = 48, n_joints: int = 15,
               guidance_scale: float = 3.0) -> torch.Tensor:
        """Full reverse diffusion: pure noise -> motion."""
        device = next(model.parameters()).device
        B = class_label.shape[0]
        x = torch.randn(B, n_frames, n_joints, 3, device=device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t, class_label, guidance_scale)
        return x