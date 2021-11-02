from typing import Optional

from ogb.graphproppred import PygGraphPropPredDataset
from pytorch_lightning import LightningDataModule
from torch_geometric.data import DataLoader, Dataset


class OGBGMolpcba(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.data_dir = data_dir

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transform = None
        self.pre_transform = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y). Pretransform is applied before saving dataset on disk."""
        PygGraphPropPredDataset(
            name="ogbg-molpcba", root=self.data_dir, pre_transform=self.pre_transform
        )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = PygGraphPropPredDataset(
                name="ogbg-molpcba", root=self.data_dir, transform=self.transform
            )
            split_idx = dataset.get_idx_split()
            self.data_train = dataset[split_idx["train"]]
            self.data_val = dataset[split_idx["valid"]]
            self.data_test = dataset[split_idx["test"]]

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
