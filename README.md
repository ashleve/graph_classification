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
python train.py --use_wandb=True --run_config MNIST_CLASSIFIER_V1
```
<br>


#### PyCharm setup
- open this repository as PyCharm project
- set project interpreter:<br> 
`Ctrl + Shift + A -> type "Project Interpreter"`
- mark folder "project" as sources root:<br>
`right click on directory -> "Mark Directory as" -> "Sources Root"`
- set terminal emulation:<br> 
`Ctrl + Shift + A -> type "Edit Configurations..." -> select "Emulate terminal in output console"`
- run training:<br>
`right click on train.py file -> "Run 'train'"`

#### VS Code setup
- TODO
<br>


## Project config parameters ([project_config.yaml](project/project_config.yaml))
Example project configuration:
```yaml
num_of_gpus: -1             <- '-1' to use all gpus available, '0' to train on cpu

loggers:
    wandb:
        project: "project_name"     <- wandb project name
        entity: "some_name"         <- wandb entity name
        log_model: True             <- set True if you want to upload ckpts to wandb automatically
        offline: False              <- set True if you want to store all data locally

callbacks:
    checkpoint:
        monitor: "val_acc"      <- name of the logged metric that determines when model is improving
        save_top_k: 1           <- save k best models (determined by above metric)
        save_last: True         <- additionaly always save model from last epoch
        mode: "max"             <- can be "max" or "min"
    early_stop:
        monitor: "val_acc"      <- name of the logged metric that determines when model is improving
        patience: 5             <- how many epochs of not improving until training stops
        mode: "max"             <- can be "max" or "min"

printing:
    progress_bar_refresh_rate: 5    <- refresh rate of training bar in terminal
    weights_summary: "top"          <- print summary of model (alternatively "full")
    profiler: False                 <- set True if you want to see execution time profiling
```
<br>


## Run config parameters ([run_configs.yaml](project/run_configs.yaml))
You can store many run configurations in this file.<br>
Example run configuration:
```yaml
MNIST_CLASSIFIER_V1:
    trainer:                                            <- lightning 'Trainer' parameters (all except 'max_epochs' are optional)
        max_epochs: 5                                       
        gradient_clip_val: 0.5                              
        accumulate_grad_batches: 3                          
        limit_train_batches: 1.0                            
    model:                                              <- all of the parameters here will be passed to 'LitModel' in 'hparams' dictionary
        model_folder: "simple_mnist_classifier"             <- name of folder from which 'lightning_module.py' (with 'LitMdodel' class) will be loaded
        lr: 0.001                                           
        weight_decay: 0.000001                              
        input_size: 784                                     
        output_size: 10                                     
        lin1_size: 256                                      
        lin2_size: 256                                      
        lin3_size: 128                                      
    dataset:                                            <- all of the parameters here will be passed to 'DataModule' in 'hparams' dictionary
        datamodule_folder: "mnist_digits_datamodule"        <- name of folder from which 'datamodule.py' (with 'DataModule' class) will be loaded
        batch_size: 256                                     
        train_val_split_ratio: 0.9                          
        num_workers: 1                                      
        pin_memory: False
    wandb:                                              <- this section is optional and can be removed
        group: ""
        tags: ["v1", "uwu"]
    resume_training:                                    <- this section is optional and can be removed if you don't want to resume training
        checkpoint_path: "path_to_checkpoint/last.ckpt"     <- path to checkpoint
        wandb_run_id: None                                  <- you can set id of Weights&Biases run that you want to resume but it's optional                        
```
<br>
