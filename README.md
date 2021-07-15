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

# create conda environment
bash setup_conda.sh
conda activate env_name
```

Train model with default configuration
```yaml
# default
python run.py

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
Coming soon...

## Results
Coming soon...
