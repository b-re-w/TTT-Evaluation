from ..base import BaseModel
from torchvision import models
import torch.nn as nn


class ResNet50(BaseModel):
    model_name = "ResNet50"

    def __init__(self, image_size: int, num_classes: int):
        super().__init__(image_size=image_size, num_classes=num_classes)
        self.model = models.resnet50(pretrained=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x, *args, **kwargs):
        return self.model(x)


class ResNet101(BaseModel):
    model_name = "ResNet101"

    def __init__(self, image_size: int, num_classes: int):
        super().__init__(image_size=image_size, num_classes=num_classes)
        self.model = models.resnet101(pretrained=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x, *args, **kwargs):
        return self.model(x)


class ResNet152(BaseModel):
    model_name = "ResNet152"

    def __init__(self, image_size: int, num_classes: int):
        super().__init__(image_size=image_size, num_classes=num_classes)
        self.model = models.resnet152(pretrained=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x, *args, **kwargs):
        return self.model(x)
