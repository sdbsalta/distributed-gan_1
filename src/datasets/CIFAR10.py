from torch import nn
import torch
from typing import Tuple
from datasets.DataPartitioner import DataPartitioner, _get_partition
from torchvision.datasets import CIFAR10
from torchvision import transforms
from typing import List, Tuple

SHAPE: Tuple[int, int, int] = (3, 32, 32)
NDF: int = 64
NGF: int = 64
Z_DIM: int = 100
NGPU: int = 1


class Partitioner(DataPartitioner):
    """
    Partition CIFAR10 dataset
    """

    def __init__(self, world_size: int, rank: int, path: str = "data/cifar10"):
        self.world_size = world_size
        self.rank = rank
        self.cifar10_train = None
        self.cifar10_test = None
        self.path = path

    def load_data(self):
        transform = transforms.Compose(
            [
                # transforms.Resize(64),
                # transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.cifar10_train = CIFAR10(root=self.path, download=True, transform=transform)
        self.cifar10_test = CIFAR10(
            root=self.path, train=False, download=True, transform=transform
        )

    def get_subset_from_indices(
        self, indices: List[int], train: bool = True
    ) -> torch.utils.data.Subset:
        if train:
            return torch.utils.data.Subset(self.cifar10_train, indices)
        return torch.utils.data.Subset(self.cifar10_test, indices)

    def get_train_partition(
        self, partition_id: int
    ) -> Tuple[torch.utils.data.Subset, int, int]:
        return _get_partition(self.world_size, partition_id, self.cifar10_train)

    def shuffle(self):
        self.cifar10_train = torch.utils.data.Subset(
            self.cifar10_train, torch.randperm(len(self.cifar10_train))
        )
        self.cifar10_test = torch.utils.data.Subset(
            self.cifar10_test, torch.randperm(len(self.cifar10_test))
        )

    def get_test_partition(
        self, partition_id: int
    ) -> Tuple[torch.utils.data.Subset, int, int]:
        return _get_partition(self.world_size, partition_id, self.cifar10_test)

    @property
    def train_dataset(self) -> torch.utils.data.Dataset:
        return self.cifar10_train

    @property
    def test_dataset(self) -> torch.utils.data.Dataset:
        return self.cifar10_test


class Discriminator(nn.Module):
    """
    https://github.com/Ksuryateja/DCGAN-CIFAR10-pytorch
    """

    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(SHAPE[0], NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(NDF, NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(NDF * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.is_cuda and NGPU > 1:
            output = nn.parallel.data_parallel(self.main, input, range(NGPU))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


class Generator(nn.Module):
    """
    https://github.com/Ksuryateja/DCGAN-CIFAR10-pytorch
    """

    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(Z_DIM, NGF * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(NGF * 2, SHAPE[0], 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (nc) x 32 x 32
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.is_cuda and NGPU > 1:
            output = nn.parallel.data_parallel(self.main, input, range(NGPU))
        else:
            output = self.main(input)
        return output
