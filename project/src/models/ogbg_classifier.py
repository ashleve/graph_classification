from pytorch_lightning.metrics.classification import Accuracy
from ogb.graphproppred import Evaluator
import torch.nn.functional as F
import pytorch_lightning as pl
import torch

# import custom architectures
from src.architectures.gcn import GCN
from src.architectures.gat import GAT


class OGBGClassifier(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.accuracy = Accuracy()

        if self.hparams.architecture == "GCN":
            self.architecture = GCN(hparams=self.hparams)
        elif self.hparams.architecture == "GAT":
            self.architecture = GAT(hparams=self.hparams)
        elif self.hparams.architecture == "GraphSAGE":
            self.architecture = None
        else:
            raise Exception("Invalid architecture name")

        # self.criterion = torch.nn.MSELoss()
        # self.criterion = torch.nn.BCEWithLogitsLoss()
        self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.9, 0.1]))

        self.evaluator = Evaluator(name=self.hparams.dataset_name)

    def forward(self, x):
        return self.architecture(x)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = self.criterion(logits, batch.y)

        # training metrics
        y_pred = torch.argmax(logits, dim=1)
        y_true = batch.y
        acc = self.accuracy(y_pred, y_true)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        # self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "y_pred": y_pred, "y_true": y_true}

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        logits = self.architecture(batch)
        loss = self.criterion(logits, batch.y)

        # training metrics
        y_pred = torch.argmax(logits, dim=1)
        y_true = batch.y
        acc = self.accuracy(y_pred, y_true)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        # self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here anything and then read it in some callback
        return {"loss": loss, "y_pred": y_pred, "y_true": y_true}

    # logic for a single test step
    def test_step(self, batch, batch_idx):
        logits = self.architecture(batch)
        loss = self.criterion(logits, batch.y)

        # training metrics
        y_pred = torch.argmax(logits, dim=1)
        y_true = batch.y
        acc = self.accuracy(y_pred, y_true)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        # self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "y_pred": y_pred, "y_true": y_true}

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        else:
            raise Exception("Invalid optimizer name")

    def training_epoch_end(self, outputs):
        y_true = torch.reshape(torch.cat([x["y_true"] for x in outputs], dim=0), (-1, 1))
        y_pred = torch.reshape(torch.cat([x["y_pred"] for x in outputs], dim=0), (-1, 1))
        result_dict = self.evaluator.eval({"y_true": y_true, "y_pred": y_pred})
        self.log("train_rocauc", result_dict["rocauc"], prog_bar=True)

    def validation_epoch_end(self, outputs):
        y_true = torch.reshape(torch.cat([x["y_true"] for x in outputs], dim=0), (-1, 1))
        y_pred = torch.reshape(torch.cat([x["y_pred"] for x in outputs], dim=0), (-1, 1))
        result_dict = self.evaluator.eval({"y_true": y_true, "y_pred": y_pred})
        self.log("val_rocauc", result_dict["rocauc"], prog_bar=True)

    def test_epoch_end(self, outputs):
        y_true = torch.reshape(torch.cat([x["y_true"] for x in outputs], dim=0), (-1, 1))
        y_pred = torch.reshape(torch.cat([x["y_pred"] for x in outputs], dim=0), (-1, 1))
        result_dict = self.evaluator.eval({"y_true": y_true, "y_pred": y_pred})
        self.log("test_rocauc", result_dict["rocauc"], prog_bar=True)
