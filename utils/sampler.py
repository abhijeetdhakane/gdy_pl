import torch.distributed as dist
from torch.utils.data import Dataset, Sampler, DistributedSampler
from typing import TypeVar, Optional, Iterator

__all__ = ["CustomInferenceSampler_v2"]

T_co = TypeVar('T_co', covariant=True)


class CustomInferenceSampler_v2(Sampler[T_co]):
    """Custom sampler for distributed inference to avoid duplicate samples."""

    def __init__(self, dataset: Dataset, num_replicas: Optional[int]=None, rank: Optional[int]=None) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

        # Compute the number of samples per replica, ensuring no extra samples are added
        self.num_samples_per_replica = len(self.dataset) // self.num_replicas
        self.remainder = len(self.dataset) % self.num_replicas

        # Compute the start and end indices for each replica
        self.start_index = self.rank * self.num_samples_per_replica + min(self.rank, self.remainder)
        self.end_index = self.start_index + self.num_samples_per_replica
        if self.rank < self.remainder:
            self.end_index += 1

    def __iter__(self) -> Iterator[T_co]:
        indices = list(range(self.start_index, self.end_index))
        return iter(indices)

    def __len__(self) -> int:
        return self.end_index - self.start_index