import torch
from torch_geometric.data import InMemoryDataset


class SuperpixelsCustomDataset(InMemoryDataset):
    """https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html"""

    def __init__(self, root):
        super().__init__(root)
        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)

    @property
    def processed_file_names(self):
        return ["data.pt"]
