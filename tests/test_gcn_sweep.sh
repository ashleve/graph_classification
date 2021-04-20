# !/bin/bash
# Test hyperparameter sweeps

# To execute:
# bash tests/sweep_tests.sh

# Method for printing test name
echo() {
  termwidth="$(tput cols)"
  padding="$(printf '%0.1s' ={1..500})"
  printf '\e[33m%*.*s %s %*.*s\n\e[0m' 0 "$(((termwidth-2-${#1})/2))" "$padding" "$1" 0 "$(((termwidth-1-${#1})/2))" "$padding"
}

# Make python hide warnings
export PYTHONWARNINGS="ignore"


echo "TEST 1"
python run.py --multirun \
experiment=GCN/gcn_mnist_sp75 \
hparams_search=gcn_hparams_search optimized_metric="val/acc_best" \
hydra.sweeper.n_trials=1 \
trainer.gpus=1 trainer.max_epochs=1 \
datamodule.num_workers=10 datamodule.pin_memory=True \
print_config=false

echo "TEST 2"
python run.py --multirun \
experiment=GCN/gcn_fashion_mnist_sp100 \
hparams_search=gcn_hparams_search \
optimized_metric="val/acc_best" \
hydra.sweeper.n_trials=1 \
trainer.gpus=1 trainer.max_epochs=1 \
datamodule.num_workers=10 datamodule.pin_memory=True \
print_config=false

echo "TEST 3"
python run.py --multirun \
experiment=GCN/gcn_cifar10_sp100 \
hparams_search=gcn_hparams_search \
optimized_metric="val/acc_best" \
hydra.sweeper.n_trials=1 \
trainer.gpus=1 trainer.max_epochs=1 \
datamodule.num_workers=10 datamodule.pin_memory=True \
print_config=false

echo "TEST 4"
python run.py --multirun \
experiment=GCN/gcn_ogbg_molhiv \
hparams_search=gcn_hparams_search \
optimized_metric="val/rocauc_best" \
hydra.sweeper.n_trials=1 \
trainer.gpus=1 trainer.max_epochs=1 \
datamodule.num_workers=10 datamodule.pin_memory=True \
print_config=false

echo "TEST 5"
python run.py --multirun \
experiment=GCN/gcn_ogbg_molpcba \
hparams_search=gcn_hparams_search \
optimized_metric="val/ap_best" \
hydra.sweeper.n_trials=1 \
trainer.gpus=1 trainer.max_epochs=1 \
datamodule.num_workers=10 datamodule.pin_memory=True \
'datamodule.batch_size=choice(128, 256, 512)' \
print_config=false
