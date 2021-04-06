python run.py --config-name config_optuna.yaml --multirun \
+experiment=GAT/gat_mnist_sp \
logger=wandb logger.wandb.group="GAT_mnist_sp75_optuna" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
datamodule.num_workers=12 datamodule.pin_memory=True \
callbacks.early_stopping.patience=5

python run.py --config-name config_optuna.yaml --multirun \
+experiment=GAT/gat_fashion_mnist_sp \
logger=wandb logger.wandb.group="GAT_fashion_mnist_sp75_optuna" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
datamodule.num_workers=12 datamodule.pin_memory=True \
callbacks.early_stopping.patience=5

python run.py --config-name config_optuna.yaml --multirun \
+experiment=GAT/gat_cifar10_sp \
logger=wandb logger.wandb.group="GAT_cifar10_sp75_optuna" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
datamodule.num_workers=12 datamodule.pin_memory=True \
callbacks.early_stopping.patience=5

python run.py --config-name config_optuna.yaml --multirun \
+experiment=GAT/gat_ogbg_molhiv \
logger=wandb logger.wandb.group="GAT_ogbg_molhiv_optuna" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
datamodule.num_workers=12 datamodule.pin_memory=True \
callbacks.early_stopping.patience=5

python run.py --config-name config_optuna.yaml --multirun \
+experiment=GAT/gat_ogbg_molpcba \
logger=wandb logger.wandb.group="GAT_ogbg_molpcba_optuna" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
datamodule.num_workers=12 datamodule.pin_memory=True \
callbacks.early_stopping.patience=5
