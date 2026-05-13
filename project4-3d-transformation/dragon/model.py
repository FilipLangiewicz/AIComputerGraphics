from pathlib import Path

import torch
import torch.nn as nn


class DragonToTeapotVectorField(nn.Module):
    def __init__(
        self,
        local_feat_dims: list[int] | None = None,
        context_feat_dims: list[int] | None = None,
        head_dims: list[int] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        local_feat_dims, context_feat_dims, head_dims = validate_dims(local_feat_dims, context_feat_dims, head_dims)

        self.point_encoder = self._mlp(3, local_feat_dims, dropout)
        d_local = local_feat_dims[-1]

        self.context_encoder = self._mlp(d_local, context_feat_dims, dropout)
        d_context = context_feat_dims[-1]

        self.flow_head = self._mlp(d_local + d_context, head_dims, dropout, out_dim=3)

    def _mlp(
        self,
        in_dim: int,
        widths: list[int],
        dropout: float,
        out_dim: int | None = None,
    ) -> nn.Sequential:

        blocks: list[nn.Module] = []
        dim = in_dim

        for w in widths:
            blocks += [nn.Linear(dim, w), nn.BatchNorm1d(w), nn.ReLU(inplace=True)]

            if dropout > 0.0:
                blocks.append(nn.Dropout(p=dropout))

            dim = w

        if out_dim is not None:
            blocks.append(nn.Linear(dim, out_dim))

        return nn.Sequential(*blocks)

    def _global_context(self, per_point: torch.Tensor) -> torch.Tensor:
        pooled, _ = per_point.max(dim=1)

        return self.context_encoder(pooled)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        B, N, _ = pts.shape

        flat = pts.reshape(B * N, 3)
        local_feat = self.point_encoder(flat)

        context = self._global_context(local_feat.view(B, N, -1))

        context_tiled = context.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        aggregated = torch.cat([local_feat, context_tiled], dim=-1)

        flow = self.flow_head(aggregated)

        return flow.view(B, N, 3)


def load_model(
    checkpoint_path: str | Path,
    local_feat_dims: list[int] | None = None,
    context_feat_dims: list[int] | None = None,
    head_dims: list[int] | None = None,
    dropout: float = 0.0,
    device: str = "cpu",
) -> DragonToTeapotVectorField:

    local_feat_dims, context_feat_dims, head_dims = validate_dims(local_feat_dims, context_feat_dims, head_dims)
    device_t = torch.device(device)

    model = DragonToTeapotVectorField(
        local_feat_dims=local_feat_dims,
        context_feat_dims=context_feat_dims,
        head_dims=head_dims,
        dropout=dropout,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device_t))
    model.to(device_t)
    model.eval()

    return model


def validate_dims(
        local_feat_dims: list[int] | None,
        context_feat_dims: list[int] | None,
        head_dims: list[int] | None
) -> tuple[list[int], list[int], list[int]]:

    if local_feat_dims is None:
        local_feat_dims = [64, 128]

    if context_feat_dims is None:
        context_feat_dims = [256, 512]

    if head_dims is None:
        head_dims = [256, 128]

    return local_feat_dims, context_feat_dims, head_dims
