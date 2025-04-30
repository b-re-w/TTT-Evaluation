from transformers import FNetModel, FNetConfig
import torch.nn as nn
import torch

from ..base import BaseModel


class FNetForVisionClassification(BaseModel):
    """Model for image classification using FNet from transformers library"""

    def __init__(self, fnet_config: FNetConfig, image_size: int, num_classes: int):
        super().__init__(image_size=image_size, num_classes=num_classes)
        self.config = fnet_config
        self.image_size = image_size
        self.patch_size = fnet_config.hidden_size // 64  # Set patch size
        self.num_patches = (image_size // self.patch_size) ** 2

        # Load FNet model from transformers
        self.model = FNetModel(fnet_config)

        # Embedding layer to convert images to patches
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=fnet_config.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # Position embeddings
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, fnet_config.hidden_size)
        )

        # Classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, fnet_config.hidden_size))

        # Classification head
        self.classifier = nn.Linear(fnet_config.hidden_size, num_classes)

    def forward(self, x, *args, **kwargs):
        batch_size = x.shape[0]

        # Convert images to patches
        # (B, C, H, W) -> (B, hidden_size, H/patch_size, W/patch_size)
        patch_embeds = self.patch_embed(x)

        # (B, hidden_size, H/patch_size, W/patch_size) -> (B, hidden_size, num_patches)
        patch_embeds = patch_embeds.flatten(2)

        # (B, hidden_size, num_patches) -> (B, num_patches, hidden_size)
        patch_embeds = patch_embeds.transpose(1, 2)

        # Add classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, patch_embeds), dim=1)

        # Add position embeddings
        embeddings = embeddings + self.position_embeddings

        # Pass through FNet model
        outputs = self.model(inputs_embeds=embeddings, return_dict=True)

        # Use the output of the classification token for classification
        logits = self.classifier(outputs.last_hidden_state[:, 0])

        return logits


class FNetVisionBase(FNetForVisionClassification):
    """FNet Vision Base model"""
    model_name = "Vision-FNet-Base_P12H12"

    def __init__(self, image_size: int, num_classes: int):
        config = FNetConfig(
            hidden_size=768,
            num_hidden_layers=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            max_position_embeddings=(image_size // 16) ** 2 + 1,
            layer_norm_eps=1e-12,
            vocab_size=1  # Not used for vision tasks but required in config
        )
        super().__init__(image_size=image_size, num_classes=num_classes, fnet_config=config)


class FNetVisionLarge(FNetForVisionClassification):
    """FNet Vision Large model"""
    model_name = "Vision-FNet-Large_P16H12"

    def __init__(self, image_size: int, num_classes: int):
        config = FNetConfig(
            hidden_size=1024,
            num_hidden_layers=24,
            intermediate_size=4096,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            max_position_embeddings=(image_size // 16) ** 2 + 1,
            layer_norm_eps=1e-12,
            vocab_size=1
        )
        super().__init__(image_size=image_size, num_classes=num_classes, fnet_config=config)
