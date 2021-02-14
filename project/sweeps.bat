@echo STARTED

call conda activate graphs

python .\train.py +experiment=GCN/mnist_superpixels_150 ^
logger=wandb logger.wandb.group="optuna_mnist_sp150_GCN" ^
trainer.gpus=1 trainer.max_epochs=1 ^
--config-name config_optuna.yaml --multirun

python .\train.py +experiment=GAT/mnist_superpixels_150 ^
logger=wandb logger.wandb.group="optuna_mnist_sp150_GAT" ^
trainer.gpus=1 trainer.max_epochs=1 ^
--config-name config_optuna.yaml --multirun

@echo FINISHED

TIMEOUT /T 10
