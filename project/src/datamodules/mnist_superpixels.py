from torch.utils.data import random_split, ConcatDataset
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataLoader
import pytorch_lightning as pl


class MNISTSuperpixelsDataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.data_dir = kwargs.get("data_dir") + "/MNIST_superpixels"

        self.train_val_test_split_ratio = kwargs.get("train_val_test_split_ratio") or [0.85, 0.10, 0.05]
        self.train_val_test_split = kwargs.get("train_val_test_split") or None

        self.batch_size = kwargs.get("batch_size") or 32
        self.num_workers = kwargs.get("num_workers") or 0
        self.pin_memory = kwargs.get("pin_memory") or False

        self.transforms = None

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y). Pretransform is applied before saving dataset on disk."""
        MNISTSuperpixels(self.data_dir, train=True, pre_transform=self.transforms)
        MNISTSuperpixels(self.data_dir, train=False, pre_transform=self.transforms)

    def setup(self, stage=None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        trainset = MNISTSuperpixels(self.data_dir, train=True, transform=self.transforms)
        testset = MNISTSuperpixels(self.data_dir, train=False, transform=self.transforms)
        dataset_full = ConcatDataset(datasets=[trainset, testset])

        if not self.train_val_test_split:
            train_length = int(len(dataset_full) * self.train_val_test_split_ratio[0])
            val_length = int(len(dataset_full) * self.train_val_test_split_ratio[1])
            test_length = len(dataset_full) - train_length - val_length
            self.train_val_test_split = [train_length, val_length, test_length]

        self.data_train, self.data_val, self.data_test = random_split(dataset_full, self.train_val_test_split)

    def train_dataloader(self):
        return DataLoader(dataset=self.data_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.data_val, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(dataset=self.data_test, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)
