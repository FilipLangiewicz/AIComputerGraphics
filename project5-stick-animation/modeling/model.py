import math
import torch
import torch.nn as nn


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device) / (half - 1)
    )
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class MotionDenoiser(nn.Module):
    def __init__(
        self,
        n_joints: int = 15,
        n_frames: int = 48,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.n_frames = n_frames
        joint_dim = n_joints * 3

        self.input_proj = nn.Linear(joint_dim, d_model)

        # time embedding: sinusoidal -> MLP -> d_model
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

        # class embedding: 0=walk, 1=jump, +1 for cfg null token
        self.class_emb = nn.Embedding(num_classes + 1, d_model)
        self.null_class_idx = num_classes  # used for classifier-free guidance

        # learned positional encoding
        self.pos_emb = nn.Parameter(torch.randn(1, n_frames, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm for stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.output_proj = nn.Linear(d_model, joint_dim)

    def forward(
        self,
        x: torch.Tensor,        # [B, T, 15, 3]
        t: torch.Tensor,        # [B] timestep indices
        class_label: torch.Tensor,  # [B] 0=walk, 1=jump, null_class_idx=uncond
    ) -> torch.Tensor:
        B, T, J, C = x.shape
        x_flat = x.reshape(B, T, J * C)             # [B, T, 45]
        tokens = self.input_proj(x_flat)             # [B, T, d_model]
        tokens = tokens + self.pos_emb

        t_emb = sinusoidal_embedding(t, tokens.shape[-1])  # [B, d_model]
        t_emb = self.time_mlp(t_emb)                       # [B, d_model]
        c_emb = self.class_emb(class_label)                # [B, d_model]

        # add time and class conditioning to every token
        cond = (t_emb + c_emb).unsqueeze(1)         # [B, 1, d_model]
        tokens = tokens + cond

        out = self.transformer(tokens)               # [B, T, d_model]
        out = self.output_proj(out)                  # [B, T, 45]
        return out.reshape(B, T, J, C)               # [B, T, 15, 3]