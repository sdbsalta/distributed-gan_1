import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from md_datasets.DataPartitioner import DataPartitioner, _get_partition
from torchvision.datasets import CelebA
from torchvision import transforms
from typing import List, Tuple

SHAPE: Tuple[int, int, int] = (3, 64, 64)
NDF: int = 64
NGF: int = 64
Z_DIM: int = 100


class Partitioner(DataPartitioner):
    """
    Partition CelebA dataset
    """

    def __init__(self, world_size: int, rank: int, path: str = "data/celeba"):
        self.world_size = world_size
        self.rank = rank
        self.celeba_train = None
        self.celeba_test = None
        self.path = path

    def load_data(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Resize((64, 64)),
            ]
        )
        self.celeba_train = CelebA(root=self.path, download=False, transform=transform)
        self.celeba_test = CelebA(
            root=self.path, split="test", download=False, transform=transform
        )

    def get_subset_from_indices(
        self, indices: List[int], train: bool = True
    ) -> torch.utils.data.Subset:
        if train:
            return torch.utils.data.Subset(self.celeba_train, indices)
        return torch.utils.data.Subset(self.celeba_test, indices)

    def get_train_partition(
        self, partition_id: int
    ) -> Tuple[torch.utils.data.Subset, int, int]:
        return _get_partition(self.world_size, partition_id, self.celeba_train)

    def shuffle(self):
        self.celeba_train = torch.utils.data.Subset(
            self.celeba_train, torch.randperm(len(self.celeba_train))
        )
        self.celeba_test = torch.utils.data.Subset(
            self.celeba_test, torch.randperm(len(self.celeba_test))
        )

    def get_test_partition(
        self, partition_id: int
    ) -> Tuple[torch.utils.data.Subset, int, int]:
        return _get_partition(self.world_size, partition_id, self.celeba_test)

    @property
    def train_dataset(self) -> torch.utils.data.Dataset:
        return self.celeba_train

    @property
    def test_dataset(self) -> torch.utils.data.Dataset:
        return self.celeba_test


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.cv1 = nn.Conv2d(
            SHAPE[0], NDF, kernel_size=4, stride=2, padding=1, bias=False
        )  # (3, 64, 64) -> (64, 32, 32)
        self.cv2 = nn.Conv2d(NDF, NDF * 2, 4, 2, 1)  # (64, 32, 32) -> (128, 16, 16)
        self.bn2 = nn.BatchNorm2d(
            NDF * 2
        )  # spatial batch norm is applied on num of channels
        self.cv3 = nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1)  # (128, 16, 16) -> (256, 8, 8)
        self.bn3 = nn.BatchNorm2d(NDF * 4)
        self.cv4 = nn.Conv2d(
            NDF * 4, NDF * 8, 4, 2, 1, bias=False
        )  # (256, 8, 8) -> (512, 4, 4)
        self.bn4 = nn.BatchNorm2d(NDF * 8)
        self.cv5 = nn.Conv2d(
            NDF * 8, 1, 4, 1, 0, bias=False
        )  # (512, 4, 4) -> (1, 1, 1)

    def forward(self, x):
        x = F.leaky_relu(self.cv1(x))
        x = F.leaky_relu(self.bn2(self.cv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.cv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.cv4(x)), 0.2, True)
        x = torch.sigmoid(self.cv5(x))
        return x.view(-1, 1).squeeze(1)


class Generator(nn.Module):
    """
    https://github.com/AKASHKADEL/dcgan-celeba/blob/master/networks.py
    """

    def __init__(self) -> None:
        super(Generator, self).__init__()

        # Input is the latent vector Z.
        self.tconv1 = nn.ConvTranspose2d(
            Z_DIM, NGF * 8, kernel_size=4, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(NGF * 8)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(NGF * 4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(NGF * 2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(NGF * 2, NGF, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(NGF)

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(NGF, SHAPE[0], 4, 2, 1, bias=False)
        # Output Dimension: (nc) x 64 x 64

    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))

        x = torch.tanh(self.tconv5(x))

        return x
