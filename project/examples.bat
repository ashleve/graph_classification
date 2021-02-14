@echo START

call conda activate graphs

python train.py trainer.max_epochs=1

python train.py trainer.gpus=1 trainer.max_epochs=1

python train.py +experiment=GCN/mnist_superpixels_150 trainer.gpus=1 trainer.min_epochs=1 trainer.max_epochs=2

python train.py +experiment=GAT/mnist_superpixels_150 trainer.gpus=1 trainer.min_epochs=1 trainer.max_epochs=2

python train.py +experiment=GCN/mnist_superpixels_75 trainer.gpus=1 trainer.max_epochs=1

python train.py +experiment=GAT/mnist_superpixels_75 trainer.gpus=1 trainer.max_epochs=1

@echo FINISHED
TIMEOUT /T 10
