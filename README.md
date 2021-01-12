<div align="center">    
 
# Graph Classification Experiments 

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  

</div>

## Description
What it does

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

pip install hydra-core --upgrade --pre
```

Next, you can train model with default configuration without logging
```bash
cd project
python train.py
```

Or you can train model with chosen logger like Weights&Biases
```yaml
# set project and entity names in project/configs/logger/wandb.yaml
wandb:
    args:
        project: "your_project_name"
        entity: "your_wandb_team_name"
```
```bash
# train model with Weights&Biases
python train.py logger=wandb.yaml
```

Or you can train model with chosen experiment config
```bash
python train.py +experiment=exp_example_simple.yaml
```

Optionally you can install project as a package with [setup.py](setup.py)
```bash
pip install -e .
```
<br>
