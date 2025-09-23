import torch
import torch.nn as nn
from typing import Tuple
from torchvision.datasets import MNIST
from torch.functional import F
from torchvision import transforms
from datasets.DataPartitioner import DataPartitioner, _get_partition
from torchvision.datasets import MNIST
from torchvision import transforms
from typing import List, Tuple


SHAPE: Tuple[int, int, int] = (1, 28, 28)
NDF: int = 64
NGF: int = 64
Z_DIM: int = 100


class Partitioner(DataPartitioner):
    """
    Partition MNIST dataset
    """

    def __init__(self, world_size: int, rank: int, path: str = "data/mnist"):
        self.world_size = world_size
        self.rank = rank
        self.mnist_train = None
        self.mnist_test = None
        self.path = path

    def get_subset_from_indices(
        self, indices: List[int], train: bool = True
    ) -> torch.utils.data.Subset:
        if train:
            return torch.utils.data.Subset(self.mnist_train, indices)
        return torch.utils.data.Subset(self.mnist_test, indices)

    def load_data(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5))]
        )
        self.mnist_train = MNIST(root=self.path, download=True, transform=transform)
        self.mnist_test = MNIST(
            root=self.path, train=False, download=True, transform=transform
        )

    def shuffle(self):
        self.mnist_train = torch.utils.data.Subset(
            self.mnist_train, torch.randperm(len(self.mnist_train))
        )
        self.mnist_test = torch.utils.data.Subset(
            self.mnist_test, torch.randperm(len(self.mnist_test))
        )

    def get_train_partition(
        self, partition_id: int
    ) -> Tuple[torch.utils.data.Subset, int, int]:
        return _get_partition(self.world_size, partition_id, self.mnist_train)

    def get_test_partition(
        self, partition_id: int
    ) -> Tuple[torch.utils.data.Subset, int, int]:
        return _get_partition(self.world_size, partition_id, self.mnist_test)

    @property
    def train_dataset(self) -> torch.utils.data.Dataset:
        return self.mnist_train

    @property
    def test_dataset(self) -> torch.utils.data.Dataset:
        return self.mnist_test


class Discriminator(nn.Module):
    """
    https://github.com/lyeoni/pytorch-mnist-GAN/blob/master/pytorch-mnist-GAN.ipynb
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(SHAPE[0] * SHAPE[1] * SHAPE[2], 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)

        return torch.sigmoid(self.fc4(x)).flatten()


class Generator(nn.Module):
    """
    https://github.com/lyeoni/pytorch-mnist-GAN/blob/master/pytorch-mnist-GAN.ipynb
    """
    def __init__(self):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(Z_DIM, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, SHAPE[0] * SHAPE[1] * SHAPE[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        output = torch.tanh(self.fc4(x))

        return output.view(-1, SHAPE[0], SHAPE[1], SHAPE[2])
