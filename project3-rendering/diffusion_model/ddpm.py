import torch
from diffusison_model.unet import UNet
from tqdm import tqdm


class DDPM:
    def __init__(self, T: int = 200, beta_start: float = 1e-4, beta_end: float = 0.02, device: str = "cuda"):
        self.T = T
        self.device = device

        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alpha_bar = alpha_bar
        self.sqrt_ab = alpha_bar.sqrt()
        self.sqrt_1mab = (1 - alpha_bar).sqrt()

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)

        ab = self.sqrt_ab[t][:, None, None, None]
        mab = self.sqrt_1mab[t][:, None, None, None]

        return ab * x0 + mab * noise, noise

    @torch.no_grad()
    def p_sample_loop(self, model: UNet, cond: torch.Tensor, shape: tuple) -> torch.Tensor:
        x = torch.randn(shape, device=self.device)

        model.eval()

        for t_val in tqdm(reversed(range(self.T)), desc="Sampling", leave=False, total=self.T):
            t = torch.full((shape[0],), t_val, device=self.device, dtype=torch.long)
            pred_noise = model(x, t, cond)

            beta = self.betas[t_val]
            alpha = 1.0 - beta
            alpha_bar = self.alpha_bar[t_val]

            coef = beta / (1 - alpha_bar).sqrt()
            mean = (x - coef * pred_noise) / alpha.sqrt()

            if t_val > 0:
                noise = torch.randn_like(x)
                x = mean + beta.sqrt() * noise
            else:
                x = mean

        return x