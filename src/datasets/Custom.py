import torch
import torch.nn as nn
from torchvision import datasets, transforms
from datasets.DataPartitioner import DataPartitioner, _get_partition
from typing import List, Tuple

SHAPE: Tuple[int, int, int] = (3, 64, 64)  # adjust if needed
NDF: int = 64
NGF: int = 64
Z_DIM: int = 100

class Partitioner(DataPartitioner):
    def __init__(self, world_size: int, rank: int, path: str = "data/custom"):
        self.world_size = world_size
        self.rank = rank
        self.custom_train = None
        self.custom_test = None
        self.path = path

    def load_data(self):
        transform = transforms.Compose([
            transforms.Resize((SHAPE[1], SHAPE[2])),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*SHAPE[0], [0.5]*SHAPE[0]),
        ])
        root = Path(self.path)
        self.custom_train = datasets.ImageFolder(root / "train", transform=transform)
        self.custom_test  = datasets.ImageFolder(root / "test",  transform=transform)

    def get_subset_from_indices(self, indices: List[int], train: bool = True) -> torch.utils.data.Subset:
        return torch.utils.data.Subset(self.custom_train, indices)

    def get_train_partition(self, partition_id: int) -> Tuple[torch.utils.data.Subset, int, int]:
        return _get_partition(self.world_size, partition_id, self.custom_train)

    def get_test_partition(self, partition_id: int) -> Tuple[torch.utils.data.Subset, int, int]:
        return _get_partition(self.world_size, partition_id, self.custom_test)

    def shuffle(self):
        self.custom_train = torch.utils.data.Subset(self.custom_train, torch.randperm(len(self.custom_train)))
        self.custom_test = torch.utils.data.Subset(self.custom_test, torch.randperm(len(self.custom_test)))

    @property
    def train_dataset(self) -> torch.utils.data.Dataset:
        return self.custom_train

    @property
    def test_dataset(self) -> torch.utils.data.Dataset:
        return self.custom_test


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(SHAPE[0], NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NDF, NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NDF * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(Z_DIM, NGF * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF * 2, SHAPE[0], 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
