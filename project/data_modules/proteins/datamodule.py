from project.data_modules.proteins.transforms import Indegree
from torch.utils.data import ConcatDataset, random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric import transforms
import pytorch_lightning as pl
import torch


class DataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()

        # hparams["data_dir"] is always automatically set to "path_to_project/data/"
        self.data_dir = hparams["data_dir"]

        self.train_val_split_ratio = hparams.get("train_val_split_ratio") or 0.9
        self.train_val_split = hparams.get("train_val_split") or None

        self.batch_size = hparams.get("batch_size") or 32
        self.num_workers = hparams.get("num_workers") or 1
        self.pin_memory = hparams.get("pin_memory") or False

        self.transforms = None

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def setup(self, stage=None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        # dataset_full = TUDataset(self.data_dir, name="PROTEINS", use_node_attr=True)
        dataset_full = TUDataset(self.data_dir, name="DD", use_node_attr=True)
        NUM_FEATURES, NUM_CLASSES = dataset_full.num_features, dataset_full.num_classes
        print('# %s: [FEATURES]-%d [NUM_CLASSES]-%d' % (dataset_full, NUM_FEATURES, NUM_CLASSES))

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