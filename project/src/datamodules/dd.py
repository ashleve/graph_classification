from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl


class DDDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, **args):
        super().__init__()

        self.data_dir = data_dir  # data_dir is specified in config.yaml

        self.train_val_split_ratio = args.get("train_val_split_ratio") or 0.9
        self.train_val_split = args.get("train_val_split") or None

        self.batch_size = args.get("batch_size") or 32
        self.num_workers = args.get("num_workers") or 0
        self.pin_memory = args.get("pin_memory") or False

        self.transforms = None

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def setup(self, stage=None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        dataset_full = TUDataset(self.data_dir, name="DD", use_node_attr=True)

        if not self.train_val_split:
            train_length = int(len(dataset_full) * self.train_val_split_ratio)
            val_length = len(dataset_full) - train_length
            self.train_val_split = [train_length, val_length]

        self.data_train, self.data_val = random_split(dataset_full, self.train_val_split)

    def train_dataloader(self):
        return DataLoader(dataset=self.data_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.data_val, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)
