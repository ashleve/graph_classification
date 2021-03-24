from typing import Any, Dict, List, Sequence, Tuple, Union

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.classification import Accuracy
from torch.optim import Optimizer

from src.architectures.gat_flexible import GAT
from src.architectures.gcn_flexible import GCN
from src.architectures.graph_sage_flexible import GraphSAGE


class SuperpixelClassifierModel(pl.LightningModule):
    """LightningModule for image classification from superpixels."""

    def __init__(self, *args, **kwargs):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        if self.hparams.architecture == "GCN":
            self.architecture = GCN(hparams=self.hparams)
        elif self.hparams.architecture == "GAT":
            self.architecture = GAT(hparams=self.hparams)
        elif self.hparams.architecture == "GraphSAGE":
            self.architecture = GraphSAGE(hparams=self.hparams)
        else:
            raise Exception("Invalid architecture name")

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

        self.metric_hist = {
            "train/acc": [],
            "val/acc": [],
            "train/loss": [],
            "val/loss": [],
        }

    def forward(self, x) -> torch.Tensor:
        return self.architecture(x)

    def step(self, batch) -> Dict[str, torch.Tensor]:
        logits = self.forward(batch)
        loss = self.criterion(logits, batch.y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, batch.y

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, preds, targets = self.step(batch)

        # log train metrics to loggers
        acc = self.train_accuracy(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, preds, targets = self.step(batch)

        # log val metrics to loggers
        acc = self.val_accuracy(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, preds, targets = self.step(batch)

        # log test metrics to loggers
        acc = self.test_accuracy(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return loss

    # [OPTIONAL METHOD]
    def training_epoch_end(self, outputs: List[Any]) -> None:
        # log best so far train acc and train loss
        self.metric_hist["train/acc"].append(self.trainer.callback_metrics["train/acc"])
        self.metric_hist["train/loss"].append(
            self.trainer.callback_metrics["train/loss"]
        )
        self.log("train/acc_best", max(self.metric_hist["train/acc"]), prog_bar=False)
        self.log("train/loss_best", min(self.metric_hist["train/loss"]), prog_bar=False)

    # [OPTIONAL METHOD]
    def validation_epoch_end(self, outputs: List[Any]) -> None:
        # log best so far val acc and val loss
        self.metric_hist["val/acc"].append(self.trainer.callback_metrics["val/acc"])
        self.metric_hist["val/loss"].append(self.trainer.callback_metrics["val/loss"])
        self.log("val/acc_best", max(self.metric_hist["val/acc"]), prog_bar=False)
        self.log("val/loss_best", min(self.metric_hist["val/loss"]), prog_bar=False)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        pass

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers

        """
        optim = hydra.utils.instantiate(
            self.hparams.optimizer, params=self.parameters()
        )
        return optim
