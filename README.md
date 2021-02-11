<div align="center">    
 
# Graph Classification Experiments 
Using PyTorch Lightning + Hydra to benchmark graph neural networks on graph classification datasets.<br>
Built with [lightning-hydra-template](https://github.com/hobogalaxy/lightning-hydra-template).
</div>

## Description
The following datasets have implemented [datamodules](project/src/datamodules) and [lightning models](project/src/models):
- Image classification from graphs of superpixels (MNIST, FashionMNIST, CIFAR10)
- [Open Graph Benchmarks](https://ogb.stanford.edu/docs/graphprop/): node property prediction (ogbg-molhiv, ogbg-molpcba, ogbg-ppa)

## How to run
First, install dependencies
```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# optionally create conda environment
conda update conda
conda env create -f conda_env.yaml -n your_env_name
conda activate your_env_name

# install requirements
pip install -r requirements.txt
```
Next, install pytorch geometric:<br>
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
<br>
<br>


Now you can train model with default configuration without logging:
```bash
cd project
python train.py
```

Or you can train model with chosen logger like Weights&Biases:
```yaml
# set project name and entity name in graph_classification/configs/logger/wandb.yaml
wandb:
    args:
        project: "your_project_name"
        entity: "your_wandb_team_name"
```
```bash
# train model with Weights&Biases
python train.py logger=wandb
```

Or you can train model with chosen experiment config:<br>
<b>(Other superpixel experiments require to firstly generate dataset with 
[superpixels_dataset_generation.ipynb](project/notebooks/superpixels_dataset_generation.ipynb))</b>
```bash
python train.py +experiment/GCN_benchmarks=gcn_mnist_superpixels
```
<br>

To execute all experiments from folder `graph_classification/configs/experiment/GCN_benchmarks/` run:
```bash
python train.py --multirun '+experiment/GCN_benchmarks=glob(*)'
```

You can override any parameter from command line like this:
```
python train.py trainer.max_epochs=20 optimizer.lr=0.0005
```

Combaining it all:
```
python train.py --multirun '+experiment/GCN_benchmarks=glob(*)' trainer.max_epochs=10 logger=wandb
```

Optionally you can install project as a package with [setup.py](setup.py):
```bash
pip install -e .
```
<br>
