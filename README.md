<div align="center">

# Graph Classification Benchmarks

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-orange?logo=pytorch"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-blueviolet"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-blue"></a>
[![](https://shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=303030)](https://github.com/hobogalaxy/lightning-hydra-template)

</div>

## Description
Using PyTorch Lightning + Hydra to benchmark graph neural networks on graph classification datasets.<br>
Built with [lightning-hydra-template](https://github.com/hobogalaxy/lightning-hydra-template).
<!--
The following datasets have implemented [datamodules](src/pl_datamodules) and [lightning models](src/pl_models):
- Image classification from graphs of superpixels (MNIST, FashionMNIST, CIFAR10)
- [Open Graph Benchmarks](https://ogb.stanford.edu/docs/graphprop/): graph property prediction (ogbg-molhiv, ogbg-molpcba, ogbg-ppa) -->

## How to run
Install dependencies
```yaml
# clone project
git clone https://github.com/hobogalaxy/graph_classification
cd graph_classification

# [OPTIONAL] create conda environment
conda env create -f conda_env_gpu.yaml -n gnn_env
conda activate gnn_env

# install requirements
pip install -r requirements.txt
```

Train model with default configuration
```yaml
python run.py
```

Train model with chosen experiment configuration
```yaml
# experiment configurations are placed in folder `configs/experiment/`
python run.py +experiment=exp_names
```

You can override any parameter from command line like this
```yaml
python run.py trainer.max_epochs=20 optimizer.lr=0.0005
```

Train on GPU
```yaml
python run.py trainer.gpus=1
```
<br>
