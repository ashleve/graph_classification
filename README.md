<div align="center">    
 
# Graph Classification Experiments

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
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

# optionally install project with setup.py
pip install -e .

# install requirements
pip install -r requirements.txt
```

Next, you can train model without logging
```bash
# train model without Weights&Biases
# choose run config from project/run_configs.yaml
cd project
python train.py --use_wandb=False --run_config MNIST_CLASSIFIER_V1
```

Or you can train model with Weights&Biases logging
```yaml
# set project and enity names in project/project_config.yaml
loggers:
    wandb:
        project: "your_project_name"
        entity: "your_wandb_username_or_team"
```
```bash
# train model with Weights&Biases
# choose run config from project/run_configs.yaml
cd project
python train.py --use_wandb=True --run_config BASELINE_PIXEL_MNIST_CLASSIFIER
```
<br>


## Run config parameters ([run_configs.yaml](project/run_configs.yaml))
You can store many run configurations in this file.<br>
Example run configuration:
```yaml
BASELINE_PIXEL_MNIST_CLASSIFIER:
    trainer:
        min_epochs: 1
        max_epochs: 3
        gradient_clip_val: 0.5
        accumulate_grad_batches: 1
        limit_train_batches: 1.0
    model:
        model_folder: "baseline_pixel_mnist_classifier"
        lr: 0.001
        weight_decay: 0.000001
        input_size: 784
        output_size: 10
        lin1_size: 256
        lin2_size: 256
        lin3_size: 128
    dataset:
        datamodule_folder: "mnist_pixels"
        batch_size: 64
        train_val_split: [60_000, 10_000]
        num_workers: 1
        pin_memory: False
    wandb:
        group: "mnist_pixels"
        tags: []
    resume_training:
        checkpoint_path: None
        wandb_run_id: None
           
```
<br>
