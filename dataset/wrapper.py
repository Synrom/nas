import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from typing import Any


class DatasetWrapper(torch.utils.data.Dataset):

  def __init__(self, dataset: torch.utils.data.Dataset, transform: Any | None = None):
    self.base_dataset = dataset
    self.transform = transform

  def __getitem__(self, idx):
    image, label = self.base_dataset[idx]
    original_image = transforms.ToTensor()(image.copy())

    if self.transform:
      image = self.transform(image)

    return original_image, image, label

  def __len__(self):
    return len(self.base_dataset)


def cifar10(train: bool = True,
            transform: Any | None = None,
            target_transform: Any | None = None,
            download: bool = True) -> DatasetWrapper:
  root = "./dataset/.download/train" if train else "./dataset/.download/test"
  dataset = CIFAR10(root=root,
                    train=train,
                    transform=None,
                    target_transform=target_transform,
                    download=download)
  return DatasetWrapper(dataset, transform)
