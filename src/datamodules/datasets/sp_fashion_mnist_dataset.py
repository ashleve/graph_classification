import os

import torch
import tqdm
from torch.utils.data import ConcatDataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import NormalizeScale, RadiusGraph, ToSLIC
from torchvision import transforms as T
from torchvision.datasets import FashionMNIST


class FashionMNISTSuperpixelsDataset(InMemoryDataset):
    def __init__(
        self, root, n_segments=100, max_num_neighbors=8, r=10, transform=None, pre_transform=None
    ):
        self.data_dir = root
        self.n_segments = n_segments
        self.base_transform = T.Compose(
            [
                T.ToTensor(),
                ToSLIC(n_segments=n_segments),
                RadiusGraph(r=r, max_num_neighbors=max_num_neighbors, loop=True),
                NormalizeScale(),
            ]
        )
        super().__init__(os.path.join(root, "FashionMNIST"), transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f"fashion_mnist_sp_{self.n_segments}.pt"]

    def download(self):
        FashionMNIST(self.data_dir, train=True, download=True, transform=self.base_transform)
        FashionMNIST(self.data_dir, train=False, download=True, transform=self.base_transform)

    def process(self):
        trainset = FashionMNIST(
            self.data_dir, train=True, download=True, transform=self.base_transform
        )
        testset = FashionMNIST(
            self.data_dir, train=False, download=True, transform=self.base_transform
        )
        dataset = ConcatDataset(datasets=[trainset, testset])

        # convert to superpixels
        print("This may take a few minutes on first run...")
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
