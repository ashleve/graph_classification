from typing import Optional, Sequence

import torch_geometric.transforms as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split
from torch_geometric.data import DataLoader, Dataset
from torch_geometric.transforms import NormalizeScale
from torchvision import transforms as T

from src.datamodules.datasets.sp_fashion_mnist_dataset import (
    FashionMNISTSuperpixelsDataset,
)


class FashionMNISTSuperpixelsDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Sequence[int] = (55_000, 5_000, 10_000),
        n_segments: int = 100,
        max_num_neighbors: int = 8,
        r: int = 10,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        """DataModule which converts FashionMNIST dataset to superpixel graphs.

        Args:
            data_dir (str): Path to data folder.
            train_val_test_split (Sequence[int]): Defaults to (55_000, 5_000, 10_000).
            n_segments (int): Number of superpixels per image.
            max_num_neighbors (int): Maximum number of edges for each node/superpixel.
            r (int): Connect node/superpixel to all nodes in range r (based on superpixel position).
            batch_size (int): Batch size
            num_workers (int): number of processes for data loading.
            pin_memory (bool): Pinning cuda memory for GPU users.

        """
        super().__init__()

        self.data_dir = data_dir
        self.train_val_test_split = train_val_test_split

        # superpixel graph parameters
        self.n_segments = n_segments
        self.max_num_neighbors = max_num_neighbors
        self.r = r

        # dataloader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.pre_transform = T.Compose(
            [
                NormalizeScale(),
            ]
        )
        self.transform = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes() -> int:
        return 10

    @property
    def num_node_features() -> int:
        return 1

    def prepare_data(self):
        """Download data if needed. Generate superpixel graphs. Apply pre-transforms.
        This method is called only from a single GPU. Do not use it to assign state (self.x = y)."""
        FashionMNISTSuperpixelsDataset(
            self.data_dir,
            n_segments=self.n_segments,
            r=self.r,
            max_num_neighbors=self.max_num_neighbors,
            pre_transform=self.pre_transform,
        )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        dataset = FashionMNISTSuperpixelsDataset(
            self.data_dir,
            n_segments=self.n_segments,
            r=self.r,
            max_num_neighbors=self.max_num_neighbors,
            pre_transform=self.pre_transform,
            transform=self.transform,
        )
        self.data_train, self.data_val, self.data_test = random_split(
            dataset, self.train_val_test_split
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
