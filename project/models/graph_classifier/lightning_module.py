from sklearn.metrics import accuracy_score
import pytorch_lightning as pl
import torch.nn.functional as F
import torch

# custom models
from models.graph_classifier.models import GCN


class LitModel(pl.LightningModule):
    """
        Simple graph classifier.
    """

    def __init__(self, hparams=None):
        super().__init__()

        if hparams:
            self.save_hyperparameters(hparams)

        self.model = GCN(hparams=self.hparams)

    def forward(self, x):
        return self.model(x)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        logits = self.model(batch)
        loss = F.nll_loss(logits, batch.y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        preds, targets = preds.cpu(), batch.y.cpu()
        acc = accuracy_score(preds, targets)
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        logits = self.model(batch)
        loss = F.nll_loss(logits, batch.y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        preds, targets = preds.cpu(), batch.y.cpu()
        acc = accuracy_score(preds, targets)
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        # we can return here anything and then read it in some callback
        return preds, targets

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
