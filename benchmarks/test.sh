python run.py --config-name config_optuna.yaml --multirun \
hydra.sweeper.optuna_config.n_trials=25 \
+experiment=GAT/gat_mnist_sp75 \
optimized_metric="val/acc_best" \
logger=wandb logger.wandb.group="GAT_mnist_sp75_optuna_clip_0.5" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
datamodule.num_workers=10 datamodule.pin_memory=True \
callbacks.early_stopping.patience=5
