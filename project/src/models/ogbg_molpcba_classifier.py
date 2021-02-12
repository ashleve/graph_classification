from ogb.graphproppred import Evaluator
import pytorch_lightning as pl
import torch
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch.nn import functional as F

# import custom architectures
from src.architectures.gcn import GCN
from src.architectures.gat import GAT


class OGBGMolpcbaClassifier(pl.LightningModule):

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
            self.architecture = None
        else:
            raise Exception("Invalid architecture name")

        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.evaluator = Evaluator(name="ogbg-molpcba")
        self.train_ap_hist = []
        self.val_ap_hist = []

    def forward(self, batch):
        batch.x = self.atom_encoder(batch.x)  # x is input atom feature
        # batch.edge_attr = self.bond_encoder(batch.edge_attr)
        return self.architecture(batch)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        y_pred = self.forward(batch)

        y_pred_nans = torch.sum(y_pred != y_pred)
        y_true_nans = torch.sum(batch.y != batch.y)
        self.log("y_pred_NaNs", y_pred_nans, reduce_fx=torch.sum, on_step=False, on_epoch=True, prog_bar=False)
        self.log("y_true_NaNs", y_true_nans, reduce_fx=torch.sum, on_step=False, on_epoch=True, prog_bar=False)
        # print("--------------------------")
        # print("y_pred_NaNs:")
        # print(y_pred_nans.shape)
        # print(y_pred_nans)
        # print("is_labeled_NaNs:")
        # print(y_true_nans.shape)
        # print(y_true_nans)
        # print("--------------------------")

        is_labeled_idx = batch.y == batch.y
        loss = self.criterion(y_pred[is_labeled_idx], batch.y.to(torch.float32)[is_labeled_idx])
        y_true = batch.y.view(y_pred.shape)

        # training metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "y_pred": y_pred, "y_true": y_true}

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        y_pred = self.forward(batch)
        is_labeled_idx = batch.y == batch.y
        loss = self.criterion(y_pred[is_labeled_idx], batch.y.to(torch.float32)[is_labeled_idx])
        y_true = batch.y.view(y_pred.shape)

        # training metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "y_pred": y_pred, "y_true": y_true}

    # logic for a single test step
    def test_step(self, batch, batch_idx):
        y_pred = self.forward(batch)
        is_labeled_idx = batch.y == batch.y
        loss = self.criterion(y_pred[is_labeled_idx], batch.y.to(torch.float32)[is_labeled_idx])
        y_true = batch.y.view(y_pred.shape)

        # training metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)

        return {"loss": loss, "y_pred": y_pred, "y_true": y_true}

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        else:
            raise Exception("Invalid optimizer name")

    def training_epoch_end(self, outputs):
        ap = self.calculate_metric(outputs)
        self.train_ap_hist.append(ap)
        self.log("train_ap", ap, prog_bar=True)
        self.log("train_ap_best", max(self.train_ap_hist), prog_bar=True)

    def validation_epoch_end(self, outputs):
        ap = self.calculate_metric(outputs)
        self.val_ap_hist.append(ap)
        self.log("val_ap", ap, prog_bar=True)
        self.log("val_ap_best", max(self.val_ap_hist), prog_bar=True)

    def test_epoch_end(self, outputs):
        self.log("test_ap", self.calculate_metric(outputs), prog_bar=False)

    def calculate_metric(self, outputs):
        y_true = torch.cat([x["y_true"] for x in outputs], dim=0)
        y_pred = torch.cat([x["y_pred"] for x in outputs], dim=0)
        result_dict = self.evaluator.eval({"y_true": y_true, "y_pred": y_pred})
        return result_dict["ap"]
