import torch
import torch.nn as nn

from pathlib import Path


class VectorFieldNet(nn.Module):
    def __init__(self, hidden: int = 256, layers: int = 6, freq: int = 6):
        super().__init__()

        self.freq = freq

        in_dim = 3 + 3 * 2 * freq

        modules = []
        prev = in_dim

        for i in range(layers):
            out = hidden
            modules.append(nn.Linear(prev, out))
            modules.append(nn.LayerNorm(out))
            modules.append(nn.GELU())
            prev = out + in_dim if (i % 2 == 1 and i < layers - 1) else out  # skip

        self.layers = nn.ModuleList()
        prev = in_dim
        self.skips = []

        for i in range(layers):
            self.layers.append(nn.Linear(prev, hidden))
            self.layers.append(nn.LayerNorm(hidden))
            self.layers.append(nn.GELU())

            if i % 2 == 1 and i < layers - 1:
                prev = hidden + in_dim
                self.skips.append(i)
            else:
                prev = hidden

        self.head = nn.Linear(prev, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def fourier(self, x: torch.Tensor) -> torch.Tensor:
        freqs = 2.0 ** torch.arange(self.freq, device=x.device, dtype=x.dtype)
        x_f = x.unsqueeze(-1) * freqs
        x_f = x_f.reshape(*x.shape[:-1], -1)

        return torch.cat([x, torch.sin(x_f), torch.cos(x_f)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.fourier(x)
        h = enc
        layer_idx = 0

        for i, layer in enumerate(self.layers):
            if i % 3 == 0:
                if layer_idx > 0 and (layer_idx - 1) in self.skips:
                    h = torch.cat([h, enc], dim=-1)

                h = layer(h)
                layer_idx += 1
            else:
                h = layer(h)

        return self.head(h)


def load_model(checkpoint_path: Path, device: str = "cpu") -> VectorFieldNet:
    device_t = torch.device(device)

    model = VectorFieldNet()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device_t))
    model.to(device_t)
    model.eval()

    return model