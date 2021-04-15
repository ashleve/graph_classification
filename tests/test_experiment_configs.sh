# !/bin/bash

# Method for printing test name
echo() {
  termwidth="$(tput cols)"
  padding="$(printf '%0.1s' ={1..500})"
  printf '\e[33m%*.*s %s %*.*s\n\e[0m' 0 "$(((termwidth-2-${#1})/2))" "$padding" "$1" 0 "$(((termwidth-1-${#1})/2))" "$padding"
}

# Make python hide warnings
export PYTHONWARNINGS="ignore"


echo "TEST 1"
python run.py +experiment=GAT/gat_cifar10_sp100 trainer.max_epochs=1

echo "TEST 2"
python run.py +experiment=GAT/gat_fashion_mnist_sp100 trainer.max_epochs=1

echo "TEST 3"
python run.py +experiment=GAT/gat_mnist_sp75 trainer.max_epochs=1

echo "TEST 4"
python run.py +experiment=GAT/gat_ogbg_molhiv trainer.max_epochs=1

echo "TEST 5"
python run.py +experiment=GAT/gat_ogbg_molpcba trainer.max_epochs=1


