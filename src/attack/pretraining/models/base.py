from torch import nn, load, save
from os import path, makedirs


class BaseModel(nn.Module):
    model_name = "BaseModel"
    dataset_name = "Dataset"

    def __init__(self, image_size: int, num_classes: int):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError("The forward method must be implemented in subclasses.")

    def save(self, save_path: str = path.join(".", "weights"), version: str = "", silence: bool = False):
        if not path.isdir(save_path):
            makedirs(save_path)
        if version:
            version = f"_{version}"
        model_id = f"{self.model_name}_{self.dataset_name}{version}"
        file_name = path.join(save_path, f"{model_id}.pt")
        save(self.state_dict(), file_name)
        if not silence: print(f"INFO: Model saved to {file_name}")

    def from_pretrained(self, version: str = "", weight_path: str = path.join(".", "weights")):
        if version:
            version = f"_{version}"
        model_id = f"{self.model_name}_{self.dataset_name}{version}"
        file_name = path.join(weight_path, f"{model_id}.pt")
        self.load_state_dict(load(file_name))
