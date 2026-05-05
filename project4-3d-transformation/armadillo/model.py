import torch
import torch.nn as nn


class VectorFieldNet(nn.Module):
    """
    PointNet-style vector field network.
    Maps each point (x,y,z) -> new (x,y,z) using local + global features.
    
    Args:
        local_hidden_dims:  list of hidden layer sizes for local feature MLP
        global_hidden_dims: list of hidden layer sizes for global feature MLP
        output_hidden_dims: list of hidden layer sizes for output MLP
        dropout:            dropout probability (0 = disabled)
    """

    def __init__(
        self,
        local_hidden_dims: list[int] = [64, 128],
        global_hidden_dims: list[int] = [256, 512],
        output_hidden_dims: list[int] = [256, 128],
        dropout: float = 0.0,
    ):
        super().__init__()

        self.local_mlp = self._build_mlp(3, local_hidden_dims, dropout)
        local_out_dim = local_hidden_dims[-1]

        self.global_mlp = self._build_mlp(local_out_dim, global_hidden_dims, dropout)
        global_out_dim = global_hidden_dims[-1]

        # input = local features + global features
        self.output_mlp = self._build_mlp(
            local_out_dim + global_out_dim, output_hidden_dims, dropout, final_out_dim=3
        )

    def _build_mlp(
        self,
        in_dim: int,
        hidden_dims: list[int],
        dropout: float,
        final_out_dim: int = None,
    ) -> nn.Sequential:
        layers = []
        current_dim = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(current_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = h
        if final_out_dim is not None:
            layers.append(nn.Linear(current_dim, final_out_dim))
        return nn.Sequential(*layers)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 3) batch of point clouds
        Returns:
            (B, N, 3) transformed point clouds
        """
        B, N, _ = points.shape

        # local features: (B*N, local_out_dim)
        x = points.view(B * N, 3)
        local_feats = self.local_mlp(x)

        # global features via max-pooling: (B, global_out_dim)
        local_feats_reshaped = local_feats.view(B, N, -1)
        global_input = local_feats_reshaped.permute(0, 2, 1)  # (B, C, N)
        global_input, _ = global_input.max(dim=2)              # (B, C)
        global_feats = self.global_mlp(global_input)           # (B, global_out_dim)

        # broadcast global features to each point
        global_expanded = global_feats.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)

        # concat local + global, predict offset
        combined = torch.cat([local_feats, global_expanded], dim=1)
        out = self.output_mlp(combined)  # (B*N, 3)

        return out.view(B, N, 3)