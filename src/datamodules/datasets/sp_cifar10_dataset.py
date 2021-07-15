import os

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torchvision.datasets import CIFAR10

from src.utils.superpixel_generation import convert_torchvision_dataset_to_superpixel_graphs


class CIFAR10SuperpixelsDataset(InMemoryDataset):
    """Dataset which converts CIFAR10 to dataset of superpixel graphs (on first run only)."""

    def __init__(
        self,
        root: str = "data/",
        num_workers: int = 8,
        n_segments: int = 100,
        **kwargs,
    ):
        self.data_dir = root
        self.num_workers = num_workers
        self.n_segments = n_segments
        self.slic_kwargs = kwargs

        super().__init__(os.path.join(root, "CIFAR10"))

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        """Dynamically generates filename for processed dataset based on superpixel parameters."""
        filename = ""
        filename += f"sp({self.n_segments})"
        for name, value in self.slic_kwargs.items():
            filename += f"_{name}({value})"
        filename += ".pt"
        return filename

    def download(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def process(self):
        trainset = CIFAR10(self.data_dir, train=True, download=True)
        testset = CIFAR10(self.data_dir, train=False, download=True)

        labels = np.concatenate((trainset.targets, testset.targets))
        images = np.concatenate((trainset.data, testset.data))

        dataset = (images, labels)

        data, slices = convert_torchvision_dataset_to_superpixel_graphs(
            dataset=dataset, desired_nodes=self.n_segments, num_workers=self.num_workers
        )

        data = Data(x=data["x"], edge_index=data["edge_index"], pos=data["pos"], y=data["y"])

        torch.save((data, slices), self.processed_paths[0])
