from torch.utils.data import DataLoader, ConcatDataset, random_split
from pytorch_lightning import LightningDataModule
from torchvision.transforms import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(LightningDataModule):
    """
    This is example of datamodule for MNIST dataset.
    To learn how to create datamodules visit:
        https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    """

    def __init__(self, data_dir, **args):
        super().__init__()

        self.data_dir = data_dir  # data_dir is specified in config.yaml

        self.batch_size = args["batch_size"]
        self.train_val_test_split = args["train_val_test_split"]
        self.num_workers = args["num_workers"]
        self.pin_memory = args["pin_memory"]

        self.transforms = transforms.ToTensor()

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        """Download data if needed."""
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        trainset = MNIST(self.data_dir, train=True, transform=self.transforms)
        testset = MNIST(self.data_dir, train=False, transform=self.transforms)
        dataset = ConcatDataset(datasets=[trainset, testset])

        self.data_train, self.data_val, self.data_test = random_split(dataset, self.train_val_test_split)

    def train_dataloader(self):
        return DataLoader(dataset=self.data_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.data_val, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(dataset=self.data_test, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)
