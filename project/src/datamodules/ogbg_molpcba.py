from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
import pytorch_lightning as pl
import torch


class OGBGMolpcba(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.data_dir = kwargs.get("data_dir")

        self.batch_size = kwargs.get("batch_size") or 32
        self.num_workers = kwargs.get("num_workers") or 0
        self.pin_memory = kwargs.get("pin_memory") or False

        self.transform = None
        self.pre_transform = None

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y). Pretransform is applied before saving dataset on disk."""
        PygGraphPropPredDataset(name="ogbg-molpcba", root=self.data_dir, pre_transform=self.pre_transform)

    def setup(self, stage=None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        dataset = PygGraphPropPredDataset(name="ogbg-molpcba", root=self.data_dir, transform=self.transform)
        split_idx = dataset.get_idx_split()
        self.data_train = dataset[split_idx["train"]]
        self.data_val = dataset[split_idx["valid"]]
        self.data_test = dataset[split_idx["test"]]

    def train_dataloader(self):
        return DataLoader(dataset=self.data_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.data_val, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.data_test, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=False)
