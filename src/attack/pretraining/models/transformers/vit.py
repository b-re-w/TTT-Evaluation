from transformers import ViTConfig, ViTModel

from ..base import BaseModel
import torch.nn as nn


class VisionTransformer(BaseModel):
    def __init__(self, config: ViTConfig, num_classes: int):
        super().__init__(image_size=config.image_size, num_classes=num_classes)
        self.config = config
        self.model = ViTModel(config=config)
        self.fc = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, x, *args, **kwargs):
        out = self.model(x)
        pooled = out.pooler_output  # [batch_size, hidden_size]
        logits = self.fc(pooled)  # [batch_size, num_classes]
        return logits


class ViTBase(VisionTransformer):
    model_name = "ViT-Base_P16H12"

    def __init__(self, image_size: int, num_classes: int):
        config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
        config.image_size = image_size
        super().__init__(config=config, num_classes=num_classes)


class ViTLarge(VisionTransformer):
    model_name = "ViT-Large_P16H12"

    def __init__(self, image_size: int, num_classes: int):
        config = ViTConfig.from_pretrained("google/vit-large-patch16-224")
        config.image_size = image_size
        super().__init__(config=config, num_classes=num_classes)
