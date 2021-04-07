<div align="center">

# Graph Classification Benchmarks
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/hobogalaxy/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>

</div>


## Description
Benchmarking graph neural networks on graph classification datasets, using PyTorch Lightning and Hydra.<br>
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
conda env create -f conda_env_gpu.yaml -n gnn
conda activate gnn

# install requirements
pip install -r requirements.txt
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
python run.py +experiment=GAT/gat_ogbg_molpcba
```

You can override any parameter from command line like this
```yaml
python run.py trainer.max_epochs=20 datamodule.batch_size=64
```


## Results
Coming soon...
