from typing import Any, List

import hydra
import torch
from ogb.graphproppred import Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from src.models.modules import gat, gcn, graph_sage


class OGBGMolhivModel(LightningModule):
    def __init__(
        self,
        architecture: str = "GCN",
        num_node_features: int = 300,
        activation: str = "prelu",
        num_conv_layers: int = 3,
        conv_size: int = 256,
        pool_method: str = "add",
        lin1_size: int = 128,
        lin2_size: int = 64,
        output_size: int = 1,
        lr: float = 0.001,
        weight_decay: float = 0,
        **kwargs
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        # init node embedding layer
        self.atom_encoder = AtomEncoder(emb_dim=self.hparams.num_node_features)
        # self.bond_encoder = BondEncoder(emb_dim=self.hparams.edge_emb_size)

        # init network architecture
        if self.hparams.architecture == "GCN":
            self.model = gcn.GCN(hparams=self.hparams)
        if self.hparams.architecture == "GAT":
            self.model = gat.GAT(hparams=self.hparams)
        if self.hparams.architecture == "GraphSAGE":
            self.model = graph_sage.GraphSAGE(hparams=self.hparams)

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
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "y_pred": y_pred, "y_true": y_true}

    def training_epoch_end(self, outputs: List[Any]):
        rocauc = self.calculate_metric(outputs)
        self.metric_hist["train/rocauc"].append(rocauc)
        self.metric_hist["train/loss"].append(self.trainer.callback_metrics["train/loss"])
        self.log("train/rocauc", rocauc, prog_bar=True)
        self.log("train/rocauc_best", max(self.metric_hist["train/rocauc"]), prog_bar=True)
        self.log("train/loss_best", min(self.metric_hist["train/loss"]), prog_bar=False)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, y_pred, y_true = self.step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "y_pred": y_pred, "y_true": y_true}

    def validation_epoch_end(self, outputs: List[Any]):
        rocauc = self.calculate_metric(outputs)
        self.metric_hist["val/rocauc"].append(rocauc)
        self.metric_hist["val/loss"].append(self.trainer.callback_metrics["val/loss"])
        self.log("val/rocauc", rocauc, prog_bar=True)
        self.log("val/rocauc_best", max(self.metric_hist["val/rocauc"]), prog_bar=True)
        self.log("val/loss_best", min(self.metric_hist["val/loss"]), prog_bar=False)

    def test_step(self, batch: Any, batch_idx: int):
        loss, y_pred, y_true = self.step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "y_pred": y_pred, "y_true": y_true}

    def test_epoch_end(self, outputs: List[Any]):
        self.log("test/rocauc", self.calculate_metric(outputs), prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

    def calculate_metric(self, outputs: List[Any]):
        y_true = torch.cat([x["y_true"] for x in outputs], dim=0)
        y_pred = torch.cat([x["y_pred"] for x in outputs], dim=0)
        result_dict = self.evaluator.eval({"y_true": y_true, "y_pred": y_pred})
        return result_dict["rocauc"]
