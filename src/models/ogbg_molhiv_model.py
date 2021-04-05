from typing import Any, List

import hydra
import torch
from ogb.graphproppred import Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from omegaconf import DictConfig
from pytorch_lightning import LightningModule


class OGBGMolhivModel(LightningModule):
    def __init__(self, architecture: DictConfig, optimizer: DictConfig):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        # initialize network architecture (e.g. GCN)
        self.atom_encoder = AtomEncoder(emb_dim=self.hparams.node_emb_size)
        self.model = hydra.utils.instantiate(self.hparams.architecture)
        # self.bond_encoder = BondEncoder(emb_dim=self.hparams.edge_emb_size)

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # metric
        self.evaluator = Evaluator(name="ogbg-molhiv")

        self.metric_hist = {
            "train/rocauc": [],
            "val/rocauc": [],
            "train/loss": [],
            "val/loss": [],
        }

    def forward(self, batch: Any):
        batch.x = self.atom_encoder(batch.x)
        # batch.edge_attr = self.bond_encoder(batch.edge_attr)
        return self.model(batch)

    def step(self, batch: Any):
        y_pred = self.forward(batch)
        loss = self.criterion(y_pred, batch.y.to(torch.float32))
        y_true = batch.y.view(y_pred.shape)
        return loss, y_pred, y_true

    def training_step(self, batch: Any, batch_idx: int):
        loss, y_pred, y_true = self.step(batch)

        # log train metrics to loggers
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "y_pred": y_pred, "y_true": y_true}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, y_pred, y_true = self.step(batch)

        # log val metrics to loggers
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "y_pred": y_pred, "y_true": y_true}

    def test_step(self, batch: Any, batch_idx: int):
        loss, y_pred, y_true = self.step(batch)

        # log test metrics to loggers
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "y_pred": y_pred, "y_true": y_true}

    def configure_optimizers(self):
        optim = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters())
        return optim

    def training_epoch_end(self, outputs: List[Any]):
        rocauc = self.calculate_metric(outputs)
        self.metric_hist["train/rocauc"].append(rocauc)
        self.metric_hist["train/loss"].append(self.trainer.callback_metrics["train/loss"])
        self.log("train/rocauc", rocauc, prog_bar=True)
        self.log("train/rocauc_best", max(self.metric_hist["train/rocauc"]), prog_bar=True)
        self.log("train/loss_best", min(self.metric_hist["train/loss"]), prog_bar=False)

    def validation_epoch_end(self, outputs: List[Any]):
        rocauc = self.calculate_metric(outputs)
        self.metric_hist["val/rocauc"].append(rocauc)
        self.metric_hist["val/loss"].append(self.trainer.callback_metrics["val/loss"])
        self.log("val/rocauc", rocauc, prog_bar=True)
        self.log("val/rocauc_best", max(self.metric_hist["val/rocauc"]), prog_bar=True)
        self.log("val/loss_best", min(self.metric_hist["val/loss"]), prog_bar=False)

    def test_epoch_end(self, outputs: List[Any]):
        self.log("test/rocauc", self.calculate_metric(outputs), prog_bar=False)

    def calculate_metric(self, outputs: List[Any]):
        y_true = torch.cat([x["y_true"] for x in outputs], dim=0)
        y_pred = torch.cat([x["y_pred"] for x in outputs], dim=0)
        result_dict = self.evaluator.eval({"y_true": y_true, "y_pred": y_pred})
        return result_dict["rocauc"]
