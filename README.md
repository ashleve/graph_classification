<div align="center">

# Graph Classification Benchmarks

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

This repository is supposed to be a place for curated, high quality benchmarks of Graph Neural Networks, implemented with PyTorch Lightning and Hydra.<br>
Only datasets big enough to provide good measures are taken into consideration.<br>
Built with [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).

### Datasets

- [Open Graph Benchmarks](https://ogb.stanford.edu/docs/graphprop/) (graph property prediction)
- Image classification from superpixels (MNIST, FashionMNIST, CIFAR10)

## How to run

Install dependencies

```yaml
# clone project
git clone https://github.com/ashleve/graph_classification
cd graph_classification

# [OPTIONAL] create conda environment
conda create -n myenv python=3.8
conda activate myenv

# install pytorch and pytorch geometric according to instructions
# https://pytorch.org/get-started/
# https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```yaml
# train on CPU
python run.py trainer.gpus=0

# train on GPU
python run.py trainer.gpus=1
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```yaml
python run.py experiment=GAT/gat_ogbg_molpcba
python run.py experiment=GraphSAGE/graphsage_mnist_sp75
python run.py experiment=GraphSAGE/graphsage_cifar10_sp100
```

You can override any parameter from command line like this

```yaml
python run.py trainer.max_epochs=20 datamodule.batch_size=64
```

## Methodology

For each experiment, we run a series of 10 random hparams runs, and 5 optimization runs, using Optuna bayesian sampler. The hyperparameter search configs are available under [configs/hparams_search](configs/hparams_search).

After finding best hyperparameters, each experiment was repeated 5 times with different random seeds. The only exception are the `ogbg-molhiv` experiments, which were repeated 10 times each (because of high varience of results).

The results were averaged and reported in the table below.

## Results

| Architecture | MNIST-sp75    | FashionMNIST-sp75 | CIFAR10-sp100 | ogbg-molhiv   | ogbg-molcpba  |
| ------------ | ------------- | ----------------- | ------------- | ------------- | ------------- |
| GCN          | 0.955 ± 0.014 | 0.835 ± 0.016     | 0.518 ± 0.007 | 0.755 ± 0.019 | 0.231 ± 0.003 |
| GIN          | 0.966 ± 0.008 | 0.861 ± 0.012     | 0.512 ± 0.020 | 0.757 ± 0.025 | 0.240 ± 0.001 |
| GAT          | 0.976 ± 0.008 | 0.889 ± 0.003     | 0.617 ± 0.005 | 0.751 ± 0.026 | 0.234 ± 0.003 |
| GraphSAGE    | 0.981 ± 0.005 | 0.897 ± 0.012     | 0.629 ± 0.012 | 0.761 ± 0.025 | 0.256 ± 0.004 |

The `+-` denotes standard deviation across all seeds.
