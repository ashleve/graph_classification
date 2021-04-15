python run.py  --config-name config_optuna.yaml --multirun \
hydra.sweeper.n_trials=20 \
experiment=GAT/gat_ogbg_molhiv \
optimized_metric="val/rocauc_best" \
logger=wandb logger.wandb.group="GAT_ogbg_molhiv_optuna_clip_1" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=15 \
datamodule.num_workers=8 datamodule.pin_memory=True \
callbacks.early_stopping.patience=1000 \
+trainer.val_check_interval=0.5 \
trainer.gradient_clip_val=1 \


python run.py  --config-name config_optuna.yaml --multirun \
hydra.sweeper.n_trials=20 \
experiment=GAT/gat_ogbg_molhiv \
optimized_metric="val/rocauc_best" \
logger=wandb logger.wandb.group="GAT_ogbg_molhiv_optuna_clip_0.5" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=15 \
datamodule.num_workers=8 datamodule.pin_memory=True \
callbacks.early_stopping.patience=1000 \
+trainer.val_check_interval=0.5 \
trainer.gradient_clip_val=0.5 \


python run.py  --config-name config_optuna.yaml --multirun \
hydra.sweeper.n_trials=20 \
experiment=GAT/gat_ogbg_molhiv \
optimized_metric="val/rocauc_best" \
logger=wandb logger.wandb.group="GAT_ogbg_molhiv_optuna_clip_0.001" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=15 \
datamodule.num_workers=8 datamodule.pin_memory=True \
callbacks.early_stopping.patience=1000 \
+trainer.val_check_interval=0.5 \
trainer.gradient_clip_val=0.001 \


