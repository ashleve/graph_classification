from pytorch_lightning.metrics.classification import Accuracy
import pytorch_lightning as pl
import torch.nn.functional as F
import torch

# import custom architectures
from src.architectures.gcn import GCN
from src.architectures.gat import GAT


class GraphClassifier(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.accuracy = Accuracy()
        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_acc_hist = []
        self.train_loss_hist = []
        self.val_acc_hist = []
        self.val_loss_hist = []

        if self.hparams.architecture == "GCN":
            self.architecture = GCN(hparams=self.hparams)
        elif self.hparams.architecture == "GAT":
            self.architecture = GAT(hparams=self.hparams)
        elif self.hparams.architecture == "GraphSAGE":
            raise Exception("Invalid architecture name")
        else:
            raise Exception("Invalid architecture name")

    def forward(self, x):
        return self.architecture(x)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        logits = self.architecture(batch)
        loss = self.criterion(logits, batch.y)

        # training metrics
        y_pred = torch.argmax(logits, dim=1)
        y_true = batch.y
        acc = self.accuracy(y_pred, y_true)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False)

        # log number of NaNs
        # y_pred_nans = torch.sum(logits != logits)
        # y_true_nans = torch.sum(y_true != y_true)
        # self.log("y_pred_NaNs", y_pred_nans, reduce_fx=torch.sum, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("y_true_NaNs", y_true_nans, reduce_fx=torch.sum, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        logits = self.architecture(batch)
        loss = self.criterion(logits, batch.y)

        # training metrics
        y_pred = torch.argmax(logits, dim=1)
        y_true = batch.y
        acc = self.accuracy(y_pred, y_true)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here anything and then read it in some callback
        return {"loss": loss, "y_pred": y_pred, "y_true": y_true}

    # logic for a single test step
    def test_step(self, batch, batch_idx):
        logits = self.architecture(batch)
        loss = self.criterion(logits, batch.y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, batch.y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        else:
            raise Exception("Invalid optimizer name")

    def training_epoch_end(self, outputs):
        self.train_acc_hist.append(self.trainer.callback_metrics["train_acc"])
        self.train_loss_hist.append(self.trainer.callback_metrics["train_loss"])
        self.log("train_acc_best", max(self.train_acc_hist), prog_bar=True)
        self.log("train_loss_best", min(self.train_acc_hist), prog_bar=False)

    def validation_epoch_end(self, outputs):
        self.val_acc_hist.append(self.trainer.callback_metrics["val_acc"])
        self.val_loss_hist.append(self.trainer.callback_metrics["val_loss"])
        self.log("val_acc_best", max(self.val_acc_hist), prog_bar=True)
        self.log("val_loss_best", min(self.val_loss_hist), prog_bar=False)

