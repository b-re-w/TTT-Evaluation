from transformers import LongformerModel, LongformerConfig
import torch.nn as nn
import torch

from ..base import BaseModel


class LongformerForVisionClassification(BaseModel):
    """Model for image classification using Longformer architecture from transformers library"""

    def __init__(self, longformer_config: LongformerConfig, image_size: int, num_classes: int):
        super().__init__(image_size=image_size, num_classes=num_classes)
        self.config = longformer_config
        self.image_size = image_size
        self.patch_size = longformer_config.hidden_size // 64  # Calculate patch size
        self.num_patches = (image_size // self.patch_size) ** 2

        # Load Longformer model from transformers
        self.model = LongformerModel(longformer_config)

        # Embedding layer to convert images to patches
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=longformer_config.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # Position embeddings
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, longformer_config.hidden_size)
        )

        # Classification token (CLS token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, longformer_config.hidden_size))

        # Classification head
        self.classifier = nn.Linear(longformer_config.hidden_size, num_classes)

    def forward(self, x, y=None, *args, **kwargs):
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

        # Create attention mask with global attention on CLS token
        attention_mask = torch.ones(batch_size, embeddings.shape[1], device=embeddings.device)
        global_attention_mask = torch.zeros_like(attention_mask)
        global_attention_mask[:, 0] = 1  # Set global attention on CLS token

        # Pass through Longformer model
        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            return_dict=True
        )

        # Use the output of the classification token for classification
        logits = self.classifier(outputs.last_hidden_state[:, 0])

        if y is not None and self.training:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, y)
            return loss

        return logits


class LongformerVisionBase(LongformerForVisionClassification):
    """Longformer Vision Base model"""
    model_name = "Vision-Longformer-Base_P12H12"

    def __init__(self, image_size: int, num_classes: int):
        config = LongformerConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_window=[8] * 12,  # Sliding window size for each layer
            max_position_embeddings=(image_size // 16) ** 2 + 1,
            layer_norm_eps=1e-12,
            vocab_size=1  # Not used for vision tasks but required in config
        )
        super().__init__(image_size=image_size, num_classes=num_classes, longformer_config=config)


class LongformerVisionLarge(LongformerForVisionClassification):
    """Longformer Vision Large model"""
    model_name = "Vision-Longformer-Large_P16H24"

    def __init__(self, image_size: int, num_classes: int):
        config = LongformerConfig(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_window=[12] * 24,  # Wider sliding windows for Large model
            max_position_embeddings=(image_size // 16) ** 2 + 1,
            layer_norm_eps=1e-12,
            vocab_size=1
        )
        super().__init__(image_size=image_size, num_classes=num_classes, longformer_config=config)
