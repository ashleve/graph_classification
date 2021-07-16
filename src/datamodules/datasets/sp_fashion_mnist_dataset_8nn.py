import os
from typing import Callable, Optional

import torch
import tqdm
from torch.utils.data import ConcatDataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import RadiusGraph, ToSLIC
from torchvision import transforms as T
from torchvision.datasets import FashionMNIST


class FashionMNISTSuperpixelsDataset(InMemoryDataset):
    """Dataset which converts FashionMNISTS to superpixel graphs (on first run only)."""

    def __init__(
        self,
        root: str = "data/",
        n_segments: int = 100,
        max_num_neighbors: int = 8,
        r: float = 10,
        loop: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        **kwargs,
    ):
        self.data_dir = root
        self.n_segments = n_segments
        self.max_num_neighbors = max_num_neighbors
        self.r = r
        self.loop = loop
        self.slic_kwargs = kwargs
        self.base_transform = T.Compose(
            [
                T.ToTensor(),
                ToSLIC(n_segments=n_segments, **kwargs),
                RadiusGraph(r=r, max_num_neighbors=max_num_neighbors, loop=loop),
            ]
        )
        super().__init__(os.path.join(root, "FashionMNIST"), transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        """Dynamically generates filename for processed dataset based on superpixel parameters."""
        filename = ""
        filename += f"sp({self.n_segments})_"
        filename += f"maxn({self.max_num_neighbors})_"
        filename += f"r({self.r})_"
        filename += f"loop({self.loop})"
        for name, value in self.slic_kwargs.items():
            filename += f"_{name}({value})"
        filename += ".pt"
        return filename

    def process(self):
        trainset = FashionMNIST(
            self.data_dir, train=True, download=True, transform=self.base_transform
        )
        testset = FashionMNIST(
            self.data_dir, train=False, download=True, transform=self.base_transform
        )
        dataset = ConcatDataset(datasets=[trainset, testset])

        # convert to superpixels
        data_list = []
        for graph, label in tqdm.tqdm(dataset, desc="Generating superpixels", colour="GREEN"):
            datapoint = graph
            datapoint.y = torch.tensor(label)
            data_list.append(datapoint)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
