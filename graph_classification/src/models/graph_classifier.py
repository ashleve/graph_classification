from pytorch_lightning.metrics.classification import Accuracy
from template_utils.initializers import load_class
import pytorch_lightning as pl
import torch.nn.functional as F
import torch

# import custom architectures
from src.architectures.gcn import GCN


class GraphClassifier(pl.LightningModule):

    def __init__(self, model_config, optimizer_config):
        super().__init__()
        hparams = {**model_config["args"], **optimizer_config["args"]}
        self.save_hyperparameters(hparams)
        self.optimizer_config = optimizer_config

        self.model = GCN(hparams=self.hparams)

        self.accuracy = Accuracy()

    def forward(self, x):
        return self.model(x)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        logits = self.model(batch)
        loss = F.nll_loss(logits, batch.y.long())

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        logits = self.model(batch)
        loss = F.nll_loss(logits, batch.y.long())

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here anything and then read it in some callback
        return {"batch_val_loss": loss, "batch_val_acc": acc, "batch_val_preds": preds, "batch_val_y": y}

    def configure_optimizers(self):
        Optimizer = load_class(self.optimizer_config["class"])
        return Optimizer(self.parameters(), **self.optimizer_config["args"])
