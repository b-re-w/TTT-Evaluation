from ..base import BaseModel
import torch.nn as nn

from einops.layers.torch import Rearrange


class MLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, dim, num_patches, token_dim, channel_dim, dropout=0.0):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            MLPBlock(num_patches, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            MLPBlock(dim, channel_dim, dropout)
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x


class MLPMixer(nn.Module):
    def __init__(
            self,
            image_size: int = 64,
            patch_size: int = 16,
            output_dim: int = 40,
            dropout: int = 0.01,
            embed_dim: int = 256,
            mixer_height: int = 8,
            token_dim: int = 128,
            channel_dim: int = 1024
    ):
        super().__init__()

        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size
        self.embed_dim = embed_dim
        self.mixer_height = mixer_height

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(self.patch_dim, embed_dim)
        )

        self.mixer_layers = nn.ModuleList([
            MixerBlock(embed_dim, self.num_patches, token_dim, channel_dim, dropout)
            for _ in range(mixer_height)
        ])

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.to_patch_embedding(x)

        for mixer_block in self.mixer_layers:
            x = mixer_block(x)

        x = self.layer_norm(x)
        x = x.mean(dim=1)  # Global average pooling

        return self.head(x)


class MLPMixerBase(BaseModel):
    model_name = "MLPMixer-Base_P16_H12"

    def __init__(self, image_size: int, num_classes: int):
        super().__init__(image_size=image_size, num_classes=num_classes)
        self.model = MLPMixer(
            image_size=image_size,
            output_dim=num_classes,
            patch_size=16,
            embed_dim=768,
            mixer_height=12,
            token_dim=384,
            channel_dim=3072
        )

    def forward(self, x, *args, **kwargs):
        return self.model(x)


class MLPMixerLarge(BaseModel):
    model_name = "MLPMixer-Large_P16_H24"

    def __init__(self, image_size: int, num_classes: int):
        super().__init__(image_size=image_size, num_classes=num_classes)
        self.model = MLPMixer(
            image_size=image_size,
            output_dim=num_classes,
            patch_size=16,
            embed_dim=1024,
            mixer_height=24,
            token_dim=512,
            channel_dim=4096
        )

    def forward(self, x, *args, **kwargs):
        return self.model(x)
