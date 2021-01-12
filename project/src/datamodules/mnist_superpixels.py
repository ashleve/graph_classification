from torch.utils.data import random_split, ConcatDataset
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataLoader
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()

        # hparams["data_dir"] is always automatically set to "path_to_project/data/"
        self.data_dir = hparams["data_dir"] + "/MNIST_superpixels"

        self.train_val_split_ratio = hparams.get("train_val_split_ratio") or 0.9
        self.train_val_split = hparams.get("train_val_split") or None

        self.batch_size = hparams.get("batch_size") or 32
        self.num_workers = hparams.get("num_workers") or 1
        self.pin_memory = hparams.get("pin_memory") or False

        self.transforms = None

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        """Download data if needed."""
        pass

    def setup(self, stage=None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        trainset = MNISTSuperpixels(self.data_dir, train=True, transform=self.transforms)
        testset = MNISTSuperpixels(self.data_dir, train=False, transform=self.transforms)
        dataset_full = ConcatDataset(datasets=[trainset, testset])

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
