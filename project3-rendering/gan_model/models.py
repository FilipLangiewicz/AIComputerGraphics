import torch
import torch.nn as nn


def weights_init(m):
    """DCGAN-style weight initialization (mean=0, std=0.02)."""
    cls = m.__class__.__name__
    if "Conv" in cls:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in cls:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """Conditional Generator: [z || c] -> 3x128x128

    FC: (noise_dim + cond_dim) -> features*8 x 4x4
    ConvT: 4->8->16->32->64->128, Tanh output
    """
    def __init__(
        self,
        noise_dim: int = 64,
        cond_dim:  int = 10,
        features:  int = 64,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.features  = features

        self.fc = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, features * 8 * 4 * 4),
            nn.ReLU(True),
        )
        self.net = nn.Sequential(
            self._up(features * 8, features * 8),
            self._up(features * 8, features * 4),
            self._up(features * 4, features * 2),
            self._up(features * 2, features),
            nn.ConvTranspose2d(features, 3, 4, 2, 1),
            nn.Tanh(),
        )

    @staticmethod
    def _up(ic, oc):
        return nn.Sequential(
            nn.ConvTranspose2d(ic, oc, 4, 2, 1, bias=False),
            nn.BatchNorm2d(oc),
            nn.ReLU(True),
        )

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        x = self.fc(x).view(-1, self.features * 8, 4, 4)
        return self.net(x)


class Discriminator(nn.Module):
    """Conditional Discriminator: (3x128x128, c) -> logit

    Conv: 128->64->32->16->8, flatten
    Condition injected after conv branch via linear projection.
    """
    def __init__(
        self,
        cond_dim: int = 10,
        features: int = 64,
    ):
        super().__init__()
        f = features

        self.img_branch = nn.Sequential(
            nn.Conv2d(3, f, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            self._down(f,     f * 2),
            self._down(f * 2, f * 4),
            self._down(f * 4, f * 8),
        )
        img_flat = f * 8 * 8 * 8

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, f * 8),
            nn.LeakyReLU(0.2, True),
        )
        self.head = nn.Sequential(
            nn.Linear(img_flat + f * 8, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),
            nn.Linear(1024, 1),
        )

    @staticmethod
    def _down(ic, oc):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 4, 2, 1, bias=False),
            nn.BatchNorm2d(oc),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, img, c):
        feat = self.img_branch(img).flatten(1)
        cond = self.cond_proj(c)
        return self.head(torch.cat([feat, cond], dim=1))


def build_models(
    noise_dim:  int = 64,
    cond_dim:   int = 10,
    features_g: int = 64,
    features_d: int = 64,
    device:     str = "cuda",
):
    G = Generator(noise_dim, cond_dim, features_g).to(device)
    D = Discriminator(cond_dim, features_d).to(device)
    G.apply(weights_init)
    D.apply(weights_init)
    print(f"Generator      params: {sum(p.numel() for p in G.parameters()):,}")
    print(f"Discriminator  params: {sum(p.numel() for p in D.parameters()):,}")
    return G, D