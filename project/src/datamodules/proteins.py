from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from sklearn.model_selection import KFold
import pytorch_lightning as pl


class ProteinsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, **args):
        super().__init__()

        self.data_dir = data_dir

        # K-fold params
        self.split_seed = args.get("split_seed") or None
        self.num_splits = args.get("num_splits") or 10
        self.fold = args.get("fold") or 1
        assert 1 <= self.fold <= self.num_splits, "incorrect fold number"

        self.batch_size = args.get("batch_size") or 32
        self.num_workers = args.get("num_workers") or 0
        self.pin_memory = args.get("pin_memory") or False

        self.transforms = None

        self.data_train = None
        self.data_val = None

    def setup(self, stage=None):
        """Load data. Set variables: self.data_train, self.data_val"""
        dataset_full = TUDataset(self.data_dir, name="PROTEINS", use_node_attr=True, transform=self.transforms)

        # choose k fold to train on
        kf = KFold(n_splits=self.num_splits, shuffle=True, random_state=self.split_seed)
        all_splits = [k for k in kf.split(dataset_full)]
        train_indexes, val_indexes = all_splits[self.fold]
        train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

        self.data_train, self.data_val = dataset_full[train_indexes], dataset_full[val_indexes]

    def train_dataloader(self):
        return DataLoader(dataset=self.data_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.data_val, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)
