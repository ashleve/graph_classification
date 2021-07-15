python run.py --multirun \
experiment=GraphSAGE/graphsage_fashion_mnist_sp100 \
logger=wandb logger.wandb.group="GraphSAGE_fashion_mnist_sp100_eval_new" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
callbacks.early_stopping.patience=5 \
datamodule.num_workers=8 datamodule.n_segments=100 \
seed=111,222,333,444,555,666


python run.py --multirun \
experiment=GraphSAGE/graphsage_fashion_mnist_sp100 \
logger=wandb logger.wandb.group="GraphSAGE_fashion_mnist_sp150_eval_new" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
callbacks.early_stopping.patience=5 \
datamodule.num_workers=8 datamodule.n_segments=150 \
seed=111,222,333,444,555,666
