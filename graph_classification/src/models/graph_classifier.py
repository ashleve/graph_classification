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

        if self.hparams.architecture == "GCN":
            self.architecture = GCN(hparams=self.hparams)
        elif self.hparams.architecture == "GAT":
            self.architecture = GAT(hparams=self.hparams)
        elif self.hparams.architecture == "GraphSAGE":
            self.architecture = None
        else:
            raise Exception("Invalid architecture name")

    def forward(self, x):
        return self.architecture(x)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        logits = self.architecture(batch)
        loss = F.nll_loss(logits, batch.y.long())

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, batch.y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        logits = self.architecture(batch)
        loss = F.nll_loss(logits, batch.y.long())

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, batch.y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here anything and then read it in some callback
        return {"batch_val_loss": loss, "batch_val_acc": acc, "batch_val_preds": preds, "batch_val_y": batch.y}

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        else:
            raise Exception("Invalid optimizer name")
