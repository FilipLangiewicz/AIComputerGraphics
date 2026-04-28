import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2

        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=device) / (half - 1))
        emb = t[:, None].float() * freqs[None, :]

        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, cond_dim: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, cond):
        h = self.norm1(F.silu(self.conv1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]

        scale, shift = self.cond_proj(F.silu(cond)).chunk(2, dim=1)

        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.norm2(F.silu(self.conv2(h)))

        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()

        self.norm = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape

        h = self.norm(x).reshape(B, C, H*W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)

        return x + h.permute(0, 2, 1).reshape(B, C, H, W)


class UNet(nn.Module):
    def __init__(self, param_dim: int = 10, base_ch: int = 64, time_emb_dim: int = 128):
        super().__init__()

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        self.cond_emb = nn.Sequential(
            nn.Linear(param_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        c = base_ch

        # Encoder
        self.enc1 = ResBlock(3,     c,   time_emb_dim, time_emb_dim)
        self.enc2 = ResBlock(c,   c*2,   time_emb_dim, time_emb_dim)
        self.enc3 = ResBlock(c*2, c*4,   time_emb_dim, time_emb_dim)
        self.down = nn.MaxPool2d(2)

        # Bottleneck
        self.mid1 = ResBlock(c * 4, c * 4, time_emb_dim, time_emb_dim)
        self.mid_attn = SelfAttention(c * 4)
        self.mid2 = ResBlock(c * 4, c * 4, time_emb_dim, time_emb_dim)

        # Decoder
        self.up3  = nn.ConvTranspose2d(c*4, c*4, 2, 2)
        self.dec3 = ResBlock(c*8, c*2, time_emb_dim, time_emb_dim)
        self.up2  = nn.ConvTranspose2d(c*2, c*2, 2, 2)
        self.dec2 = ResBlock(c*4, c,   time_emb_dim, time_emb_dim)
        self.up1  = nn.ConvTranspose2d(c,   c,   2, 2)
        self.dec1 = ResBlock(c*2, c,   time_emb_dim, time_emb_dim)

        self.out  = nn.Conv2d(c, 3, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        t_emb = self.time_emb(t)
        c_emb = self.cond_emb(cond)

        e1 = self.enc1(x, t_emb, c_emb)
        e2 = self.enc2(self.down(e1), t_emb, c_emb)
        e3 = self.enc3(self.down(e2), t_emb, c_emb)

        m = self.mid1(self.down(e3), t_emb, c_emb)
        m = self.mid_attn(m)
        m = self.mid2(m, t_emb, c_emb)

        d3 = self.dec3(torch.cat([self.up3(m), e3], 1), t_emb, c_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1), t_emb, c_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1), t_emb, c_emb)

        return self.out(d1)
