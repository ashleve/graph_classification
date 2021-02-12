# pytorch lightning imports
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
import pytorch_lightning as pl
import torch

# hydra imports
from omegaconf import DictConfig
import hydra

# normal imports
from typing import List

# template utils imports
from src.utils import template_utils as utils


def train(config):
    # Set seed for random number generators in pytorch, numpy, python.random
    if "seed" in config:
        pl.seed_everything(config["seed"])

    # Init PyTorch Lightning model ⚡
    model: LightningModule = hydra.utils.instantiate(config["model"])

    # Init PyTorch Lightning datamodule ⚡
    datamodule: LightningDataModule = hydra.utils.instantiate(config["datamodule"])
    datamodule.prepare_data()
    datamodule.setup()

    # dt = datamodule.train_dataloader()
    # dv = datamodule.val_dataloader()
    # dtest = datamodule.test_dataloader()
    #
    # total_sum = 0
    # total_sum2 = 0
    # for batch in dt:
    #     x, y = batch.x, batch.y
    #     total_sum += torch.sum(y == 1)
    #     total_sum2 += torch.sum(y == 0)
    #
    # print()
    # print(total_sum, total_sum2, len(datamodule.data_train))
    # print(total_sum / len(datamodule.data_train))
    # print()
    #
    # total_sum = 0
    # total_sum2 = 0
    # for batch in dv:
    #     x, y = batch.x, batch.y
    #     total_sum += torch.sum(y == 1)
    #     total_sum2 += torch.sum(y == 0)
    #
    # print()
    # print(total_sum, total_sum2, len(datamodule.data_val))
    # print(total_sum / len(datamodule.data_val))
    # print()
    #
    # total_sum = 0
    # total_sum2 = 0
    # for batch in dtest:
    #     x, y = batch.x, batch.y
    #     total_sum += torch.sum(y == 1)
    #     total_sum2 += torch.sum(y == 0)
    #
    # print()
    # print(total_sum, total_sum2, len(datamodule.data_test))
    # print(total_sum / len(datamodule.data_test))
    # print()

    # Init PyTorch Lightning callbacks ⚡
    callbacks: List[Callback] = [
        hydra.utils.instantiate(callback_conf)
        for callback_name, callback_conf in config["callbacks"].items()
        if "_target_" in callback_conf  # ignore callback conf if there's no target
    ] if "callbacks" in config else []

    # Init PyTorch Lightning loggers ⚡
    loggers: List[LightningLoggerBase] = [
        hydra.utils.instantiate(logger_conf)
        for logger_name, logger_conf in config["logger"].items()
        if "_target_" in logger_conf   # ignore logger conf if there's no target
    ] if "logger" in config else []

    # Init PyTorch Lightning trainer ⚡
    trainer: Trainer = hydra.utils.instantiate(config["trainer"], callbacks=callbacks, logger=loggers)

    # Magic
    utils.extras(config, model, datamodule, callbacks, loggers, trainer)

    trainer.test(model=model, datamodule=datamodule)

    # Train the model
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set after training
    trainer.test()

    # Finish run
    utils.finish()

    # Return best metric score for optuna
    return trainer.callback_metrics["val_rocauc_best"]


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    utils.print_config(config["model"])
    utils.print_config(config["datamodule"])
    utils.print_config(config["trainer"])
    metric = train(config)
    return metric


if __name__ == "__main__":
    main()
