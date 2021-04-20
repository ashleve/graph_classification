python run.py --multirun \
experiment=GraphSAGE/graphsage_mnist_sp75 \
hparams_search=graphsage_hparams_search optimized_metric="val/acc_best" \
hydra.sweeper.n_trials=15 \
logger=wandb logger.wandb.group="GraphSAGE_mnist_sp75_optuna" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
callbacks.early_stopping.patience=5 \
datamodule.num_workers=10 datamodule.pin_memory=True


python run.py --multirun \
experiment=GraphSAGE/graphsage_fashion_mnist_sp100 \
hparams_search=graphsage_hparams_search \
optimized_metric="val/acc_best" \
hydra.sweeper.n_trials=15 \
logger=wandb logger.wandb.group="GraphSAGE_fashion_mnist_sp100_optuna" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
callbacks.early_stopping.patience=5 \
datamodule.num_workers=10 datamodule.pin_memory=True


python run.py --multirun \
experiment=GraphSAGE/graphsage_cifar10_sp100 \
hparams_search=graphsage_hparams_search \
optimized_metric="val/acc_best" \
hydra.sweeper.n_trials=15 \
logger=wandb logger.wandb.group="GraphSAGE_cifar10_sp100_optuna" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
callbacks.early_stopping.patience=5 \
datamodule.num_workers=10 datamodule.pin_memory=True


python run.py --multirun \
experiment=GraphSAGE/graphsage_ogbg_molhiv \
hparams_search=graphsage_hparams_search \
optimized_metric="val/rocauc_best" \
hydra.sweeper.n_trials=15 \
logger=wandb logger.wandb.group="GraphSAGE_ogbg_molhiv_optuna" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
callbacks.early_stopping.patience=5 \
datamodule.num_workers=10 datamodule.pin_memory=True


python run.py --multirun \
experiment=GraphSAGE/graphsage_ogbg_molpcba \
hparams_search=graphsage_hparams_search \
optimized_metric="val/ap_best" \
'datamodule.batch_size=choice(256, 512)' \
hydra.sweeper.n_trials=15 \
logger=wandb logger.wandb.group="GraphSAGE_ogbg_molpcba_optuna" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
callbacks.early_stopping.patience=5 \
datamodule.num_workers=10 datamodule.pin_memory=True
