from typing import List, Tuple
import torch.utils.data


# Abstract class for data partitioning
class DataPartitioner:
    """
    Abstract class for data partitioning

    From a given dataset, partition the data into subsets for each worker
    """

    def __init__(self, world_size: int, rank: int):
        """
        Initialize the data partitioner, we need to know the world size and rank of the worker
        """
        raise NotImplementedError
    
    def get_subset_from_indices(self, indices: List[int], train: bool = True) -> torch.utils.data.Subset:
        """
        Get a subset of the data from a list of indices
        """
        raise NotImplementedError

    def load_data(self) -> None:
        """
        Load the data into memory
        """
        raise NotImplementedError
    
    def shuffle(self) -> None:
        """
        Shuffle the data
        """
        raise NotImplementedError

    def get_train_partition(
        self, partition_id: int
    ) -> Tuple[torch.utils.data.Subset, int, int]:
        """
        Get a train partition of the data for a worker
        """
        raise NotImplementedError

    def get_test_partition(
        self, partition_id: int
    ) -> Tuple[torch.utils.data.Subset, int, int]:
        """
        Get a test partition of the data for a worker
        """
        raise NotImplementedError

    @property
    def train_dataset(self) -> torch.utils.data.Dataset:
        raise NotImplementedError

    @property
    def test_dataset(self) -> torch.utils.data.Dataset:
        raise NotImplementedError


def _get_partition(
    world_size: int, partition_id: int, dataset: torch.utils.data.Dataset
) -> Tuple[torch.utils.data.Subset, int, int]:
    size = len(dataset)
    size = 1000
    length = size // world_size
    start = partition_id * length
    end = start + length

    # If the last partition is not divisible by the world size, assign the remaining data to the last partition
    next_len = size - end
    if next_len > 0 and partition_id == world_size - 1:
        end += next_len

    return torch.utils.data.Subset(dataset, range(start, end)), start, end
