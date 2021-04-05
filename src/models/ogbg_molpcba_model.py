import pytorch_lightning as pl
import torch
from ogb.graphproppred import Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from src.architectures.gat_flexible import GAT

# import custom architectures
from src.architectures.gcn_flexible import GCN
from src.architectures.graph_sage_flexible import GraphSAGE


class OGBGMolpcbaModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.atom_encoder = AtomEncoder(emb_dim=self.hparams.node_emb_size)
        # self.bond_encoder = BondEncoder(emb_dim=self.hparams.edge_emb_size)

        if self.hparams.architecture == "GCN":
            self.architecture = GCN(hparams=self.hparams)
        elif self.hparams.architecture == "GAT":
            self.architecture = GAT(hparams=self.hparams)
        elif self.hparams.architecture == "GraphSAGE":
            self.architecture = GraphSAGE(hparams=self.hparams)
        else:
            raise Exception("Invalid architecture name")

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.evaluator = Evaluator(name="ogbg-molpcba")

        self.train_ap_hist = []
        self.train_loss_hist = []
        self.val_ap_hist = []
        self.val_loss_hist = []

    def forward(self, batch):
        batch.x = self.atom_encoder(batch.x)  # x is input atom feature
        # batch.edge_attr = self.bond_encoder(batch.edge_attr)
        return self.architecture(batch)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)
        is_labeled_idx = batch.y == batch.y
        loss = self.criterion(logits[is_labeled_idx], batch.y.to(torch.float32)[is_labeled_idx])

        # training metrics
        y_pred = logits
        y_true = batch.y.view(y_pred.shape)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        # log number of NaNs
        y_pred_nans = torch.sum(logits != logits)
        y_true_nans = torch.sum(batch.y != batch.y)
        self.log(
            "y_pred_NaNs",
            y_pred_nans,
            reduce_fx=torch.sum,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "y_true_NaNs",
            y_true_nans,
            reduce_fx=torch.sum,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        return {"loss": loss, "y_pred": y_pred, "y_true": y_true}

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        is_labeled_idx = batch.y == batch.y
        loss = self.criterion(logits[is_labeled_idx], batch.y.to(torch.float32)[is_labeled_idx])

        # training metrics
        y_pred = logits
        y_true = batch.y.view(y_pred.shape)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "y_pred": y_pred, "y_true": y_true}

    # logic for a single test step
    def test_step(self, batch, batch_idx):
        logits = self.forward(batch)
        is_labeled_idx = batch.y == batch.y
        loss = self.criterion(logits[is_labeled_idx], batch.y.to(torch.float32)[is_labeled_idx])

        # training metrics
        y_pred = logits
        y_true = batch.y.view(y_pred.shape)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "y_pred": y_pred, "y_true": y_true}

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            return torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise Exception("Invalid optimizer name")

    def training_epoch_end(self, outputs):
        ap = self.calculate_metric(outputs)
        self.train_ap_hist.append(ap)
        self.train_loss_hist.append(self.trainer.callback_metrics["train_loss"])
        self.log("train_ap", ap, prog_bar=True)
        self.log("train_ap_best", max(self.train_ap_hist), prog_bar=True)
        self.log("train_loss_best", min(self.train_loss_hist), prog_bar=False)

    def validation_epoch_end(self, outputs):
        ap = self.calculate_metric(outputs)
        self.val_ap_hist.append(ap)
        self.val_loss_hist.append(self.trainer.callback_metrics["val_loss"])
        self.log("val_ap", ap, prog_bar=True)
        self.log("val_ap_best", max(self.val_ap_hist), prog_bar=True)
        self.log("val_loss_best", min(self.val_loss_hist), prog_bar=False)

    def test_epoch_end(self, outputs):
        self.log("test_ap", self.calculate_metric(outputs), prog_bar=False)

    def calculate_metric(self, outputs):
        y_true = torch.cat([x["y_true"] for x in outputs], dim=0)
        y_pred = torch.cat([x["y_pred"] for x in outputs], dim=0)
        result_dict = self.evaluator.eval({"y_true": y_true, "y_pred": y_pred})
        return result_dict["ap"]