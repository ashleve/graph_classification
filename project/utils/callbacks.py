from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pytorch_lightning as pl
from shutil import copy
import torch
import wandb
import os


class ExampleCallback(pl.Callback):
    def __init__(self):
        pass

    def on_init_start(self, trainer):
        print('Starting to initialize trainer!')

    def on_init_end(self, trainer):
        print('Trainer is initialized now.')

    def on_train_end(self, trainer, pl_module):
        print('Do something when training ends.')


class UnfreezeModelCallback(pl.Callback):
    """
        Unfreeze model after a few epochs.
    """
    def __init__(self, wait_epochs=5):
        self.wait_epochs = wait_epochs

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == self.wait_epochs:
            for param in pl_module.model.model.parameters():
                param.requires_grad = True


class MetricsHeatmapLoggerCallback(pl.Callback):
    """
        Generate f1, precision and recall heatmap from validation epoch outputs.
        Expects validation step to return predictions and targets.
        Works only for single label classification!
    """
    def __init__(self, class_names):
        self.class_names = class_names
        self.preds = []
        self.targets = []
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Gather data from single batch."""
        if self.ready:
            preds, targets = outputs
            self.preds.append(preds)
            self.targets.append(targets)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate f1, precision and recall heatmap."""
        if self.ready:
            self.preds = torch.cat(self.preds)
            self.targets = torch.cat(self.targets)
            f1 = f1_score(self.preds, self.targets, average=None)
            r = recall_score(self.preds, self.targets, average=None)
            p = precision_score(self.preds, self.targets, average=None)

            trainer.logger.experiment.log({
                f"f1_p_r_heatmap_{trainer.current_epoch}": wandb.plots.HeatMap(
                    x_labels=self.class_names,
                    y_labels=["f1", "precision", "recall"],
                    matrix_values=[f1, p, r],
                    show_text=True,
                )}, commit=False)

            self.preds = []
            self.targets = []


class ConfusionMatrixLoggerCallback(pl.Callback):
    """
        Generate Confusion Matrix.
        Expects validation step to return predictions and targets.
        Works only for single label classification!
    """
    def __init__(self, class_names):
        self.class_names = class_names
        self.preds = []
        self.targets = []
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Gather data from single batch."""
        if self.ready:
            preds, targets = outputs
            self.preds.append(preds)
            self.targets.append(targets)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate f1, precision and recall heatmap."""
        if self.ready:
            self.preds = torch.cat(self.preds).tolist()
            self.targets = torch.cat(self.targets).tolist()

            if not trainer or not trainer.logger or not trainer.logger.experiment:
                self.preds = []
                self.targets = []
                return

            trainer.logger.experiment.log({
                f"conf_mat{trainer.current_epoch}": wandb.plot.confusion_matrix(
                    self.preds,
                    self.targets,
                    class_names=self.class_names)
            }, commit=False)

            self.preds = []
            self.targets = []


class SaveCodeToWandbCallback(pl.Callback):
    """
        Upload specified code files to wandb at the beginning of the run.
    """
    def __init__(self, base_dir, wandb_save_dir, run_config):
        self.base_dir = base_dir
        self.wandb_save_dir = wandb_save_dir
        self.model_folder = run_config["model"]["model_folder"]
        self.datamodule_folder = run_config["dataset"]["datamodule"]
        self.additional_files_to_be_saved = [  # paths should be relative to base_dir
            "utils/callbacks.py",
            "utils/init_utils.py",
            "train.py",
            "project_config.yaml",
            "run_configs.yaml",
        ]

    def on_sanity_check_end(self, trainer, pl_module):
        """Upload files when all validation sanity checks end."""
        # upload additional files
        for file in self.additional_files_to_be_saved:
            src_path = os.path.join(self.base_dir, file)
            dst_path = os.path.join(self.wandb_save_dir, "code", file)

            path, filename = os.path.split(dst_path)
            if not os.path.exists(path):
                os.makedirs(path)

            copy(src_path, dst_path)
            wandb.save(dst_path, base_path=self.wandb_save_dir)  # this line is only to make upload immediate

        # upload model files
        for filename in os.listdir(os.path.join(self.base_dir, "models", self.model_folder)):

            if filename.endswith('.py'):
                src_path = os.path.join(self.base_dir, "models", self.model_folder, filename)
                dst_path = os.path.join(self.wandb_save_dir, "code", "models", self.model_folder, filename)

                path, filename = os.path.split(dst_path)
                if not os.path.exists(path):
                    os.makedirs(path)

                copy(src_path, dst_path)
                wandb.save(dst_path, base_path=self.wandb_save_dir)  # this line is only to make upload immediate

        # upload datamodule files
        for filename in os.listdir(os.path.join(self.base_dir, "data_modules", self.datamodule_folder)):

            if filename.endswith('.py'):
                src_path = os.path.join(self.base_dir, "data_modules", self.datamodule_folder, filename)
                dst_path = os.path.join(self.wandb_save_dir, "code", "data_modules", self.datamodule_folder, filename)

                path, filename = os.path.split(dst_path)
                if not os.path.exists(path):
                    os.makedirs(path)

                copy(src_path, dst_path)
                wandb.save(dst_path, base_path=self.wandb_save_dir)  # this line is only to make upload immediate
