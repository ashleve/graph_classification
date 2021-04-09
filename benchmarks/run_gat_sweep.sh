python run.py --config-name config_optuna.yaml --multirun \
hydra.sweeper.optuna_config.n_trials=25 \
+experiment=GAT/gat_mnist_sp75 \
optimized_metric="val/acc_best" \
logger=wandb logger.wandb.group="GAT_mnist_sp75_optuna" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
datamodule.num_workers=10 datamodule.pin_memory=True \
callbacks.early_stopping.patience=5 \
# +trainer.amp_backend="apex" +trainer.precision=16 +trainer.amp_level="O2"

python run.py --config-name config_optuna.yaml --multirun \
+experiment=GAT/gat_fashion_mnist_sp100 \
optimized_metric="val/acc_best" \
logger=wandb logger.wandb.group="GAT_fashion_mnist_sp100_optuna" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
datamodule.num_workers=10 datamodule.pin_memory=True \
callbacks.early_stopping.patience=5 \

python run.py --config-name config_optuna.yaml --multirun \
hydra.sweeper.optuna_config.n_trials=25 \
+experiment=GAT/gat_cifar10_sp100 \
optimized_metric="val/acc_best" \
logger=wandb logger.wandb.group="GAT_cifar10_sp100_optuna" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
datamodule.num_workers=10 datamodule.pin_memory=True \
callbacks.early_stopping.patience=5 \

python run.py --config-name config_optuna.yaml --multirun \
hydra.sweeper.optuna_config.n_trials=25 \
+experiment=GAT/gat_ogbg_molhiv \
optimized_metric="val/rocauc_best" \
logger=wandb logger.wandb.group="GAT_ogbg_molhiv_optuna" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
datamodule.num_workers=10 datamodule.pin_memory=True \
callbacks.early_stopping.patience=5 \

python run.py --config-name config_optuna.yaml --multirun \
hydra.sweeper.optuna_config.n_trials=25 \
+experiment=GAT/gat_ogbg_molpcba \
optimized_metric="val/ap_best" \
logger=wandb logger.wandb.group="GAT_ogbg_molpcba_optuna" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
datamodule.num_workers=10 datamodule.pin_memory=True \
callbacks.early_stopping.patience=5 \
