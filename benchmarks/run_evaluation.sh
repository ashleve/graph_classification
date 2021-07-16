python run.py --multirun \
experiment=GAT/gat_mnist_sp75 seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="GAT_mnist_sp75_eval" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
callbacks.early_stopping.patience=5 \
datamodule.num_workers=8


python run.py --multirun \
experiment=GAT/gat_fashion_mnist_sp100 seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="GAT_fashion_mnist_sp100_eval" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
callbacks.early_stopping.patience=5 \
datamodule.num_workers=8


python run.py --multirun \
experiment=GAT/gat_cifar10_sp100 seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="GAT_cifar10_sp100_eval" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
callbacks.early_stopping.patience=5 \
datamodule.num_workers=8


python run.py --multirun \
experiment=GAT/gat_ogbg_molhiv seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="GAT_ogbg_molhiv_eval" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
callbacks.early_stopping.patience=5 \
datamodule.num_workers=8


python run.py --multirun \
experiment=GAT/gat_ogbg_molpcba seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="GAT_ogbg_molpcba_eval" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
callbacks.early_stopping.patience=5 \
datamodule.num_workers=8





python run.py --multirun \
experiment=GraphSAGE/graphsage_mnist_sp75 seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="GraphSAGE_mnist_sp75_eval" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
callbacks.early_stopping.patience=5 \
datamodule.num_workers=8


python run.py --multirun \
experiment=GraphSAGE/graphsage_fashion_mnist_sp100 seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="GraphSAGE_fashion_mnist_sp100_eval" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
callbacks.early_stopping.patience=5 \
datamodule.num_workers=8


python run.py --multirun \
experiment=GraphSAGE/graphsage_cifar10_sp100 seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="GraphSAGE_cifar10_sp100_eval" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
callbacks.early_stopping.patience=5 \
datamodule.num_workers=8


python run.py --multirun \
experiment=GraphSAGE/graphsage_ogbg_molhiv seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="GraphSAGE_ogbg_molhiv_eval" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
callbacks.early_stopping.patience=5 \
datamodule.num_workers=8


python run.py --multirun \
experiment=GraphSAGE/graphsage_ogbg_molpcba seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="GraphSAGE_ogbg_molpcba_eval" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
callbacks.early_stopping.patience=5 \
datamodule.num_workers=8






python run.py --multirun \
experiment=GCN/gcn_mnist_sp75 seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="GCN_mnist_sp75_eval" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
callbacks.early_stopping.patience=5 \
datamodule.num_workers=8


python run.py --multirun \
experiment=GCN/gcn_fashion_mnist_sp100 seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="GCN_fashion_mnist_sp100_eval" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
callbacks.early_stopping.patience=5 \
datamodule.num_workers=8


python run.py --multirun \
experiment=GCN/gcn_cifar10_sp100 seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="GCN_cifar10_sp100_eval" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
callbacks.early_stopping.patience=5 \
datamodule.num_workers=8


python run.py --multirun \
experiment=GCN/gcn_ogbg_molhiv seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="GCN_ogbg_molhiv_eval" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
callbacks.early_stopping.patience=5 \
datamodule.num_workers=8


python run.py --multirun \
experiment=GCN/gcn_ogbg_molpcba seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="GCN_ogbg_molpcba_eval" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
callbacks.early_stopping.patience=5 \
datamodule.num_workers=8






python run.py --multirun \
experiment=GIN/gin_mnist_sp75 seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="GIN_mnist_sp75_eval" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
callbacks.early_stopping.patience=5 \
datamodule.num_workers=8


python run.py --multirun \
experiment=GIN/gin_fashion_mnist_sp100 seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="GIN_fashion_mnist_sp100_eval" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
callbacks.early_stopping.patience=5 \
datamodule.num_workers=8


python run.py --multirun \
experiment=GIN/gin_cifar10_sp100 seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="GIN_cifar10_sp100_eval" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
callbacks.early_stopping.patience=5 \
datamodule.num_workers=8


python run.py --multirun \
experiment=GIN/gin_ogbg_molhiv seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="GIN_ogbg_molhiv_eval" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
callbacks.early_stopping.patience=5 \
datamodule.num_workers=8


python run.py --multirun \
experiment=GIN/gin_ogbg_molpcba seed=111,222,333,444,555 \
logger=wandb logger.wandb.group="GIN_ogbg_molpcba_eval" \
trainer.gpus=1 trainer.min_epochs=10 trainer.max_epochs=100 \
callbacks.early_stopping.patience=5 \
datamodule.num_workers=8
