import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.res  = ResBlock(out_channels)
        self.pool = nn.MaxPool2d(2)

        # 1x1 conv to match dimensions for residual shortcut
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        out = self.res(out) + self.shortcut(x)  # block-level residual
        return self.pool(out), out              # (downsampled, skip)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.res  = ResBlock(out_channels)

        self.shortcut = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x   = self.up(x)
        x   = torch.cat([x, skip], dim=1)
        out = self.conv(x)
        out = self.res(out) + self.shortcut(x)
        return out


class ResUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 features: list[int] = [64, 128, 256, 512]):
        super().__init__()

        self.encoders = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(EncoderBlock(ch, f))
            ch = f

        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch, ch * 2, kernel_size=3, padding=1, bias=False),
            ResBlock(ch * 2),
        )
        ch = ch * 2

        self.decoders = nn.ModuleList()
        for f in reversed(features):
            self.decoders.append(DecoderBlock(ch, f, f))
            ch = f

        self.head = nn.Sequential(
            nn.Conv2d(ch, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.head(x)