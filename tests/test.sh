python run.py --config-name config_optuna.yaml --multirun \
hydra.sweeper.n_trials=15 \
experiment=GAT/gat_mnist_sp75 \
optimized_metric="val/acc_best" \
logger=wandb logger.wandb.group="GAT_mnist_sp75_optuna_clip_none_xavier" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=10 \
datamodule.num_workers=8 datamodule.pin_memory=True \
callbacks.early_stopping.patience=1000 \
+trainer.val_check_interval=0.5 \
trainer.gradient_clip_val=0 \


# python run.py --config-name config_optuna.yaml --multirun \
# hydra.sweeper.n_trials=30 \
# experiment=GAT/gat_fashion_mnist_sp100 \
# optimized_metric="val/acc_best" \
# logger=wandb logger.wandb.group="GAT_fashion_mnist_sp100_optuna_clip_1" \
# trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=15 \
# datamodule.num_workers=8 datamodule.pin_memory=True \
# callbacks.early_stopping.patience=1000 \
# +trainer.val_check_interval=0.5 \
# trainer.gradient_clip_val=1 \


# python run.py --config-name config_optuna.yaml --multirun \
# hydra.sweeper.n_trials=30 \
# experiment=GAT/gat_fashion_mnist_sp100 \
# optimized_metric="val/acc_best" \
# logger=wandb logger.wandb.group="GAT_fashion_mnist_sp100_optuna_clip_0.1" \
# trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=15 \
# datamodule.num_workers=8 datamodule.pin_memory=True \
# callbacks.early_stopping.patience=1000 \
# +trainer.val_check_interval=0.5 \
# trainer.gradient_clip_val=0.1 \


# python run.py --config-name config_optuna.yaml --multirun \
# hydra.sweeper.n_trials=30 \
# experiment=GAT/gat_fashion_mnist_sp100 \
# optimized_metric="val/acc_best" \
# logger=wandb logger.wandb.group="GAT_fashion_mnist_sp100_optuna_clip_0.001" \
# trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=15 \
# datamodule.num_workers=8 datamodule.pin_memory=True \
# callbacks.early_stopping.patience=1000 \
# +trainer.val_check_interval=0.5 \
# trainer.gradient_clip_val=0.001 \


# python run.py experiment=GAT/gat_mnist_sp75