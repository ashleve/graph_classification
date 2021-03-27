pip install -r requirements.txt

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-geometric

# install tab completion for this project
# RUN_PATH=$(realpath run.py)
# echo "eval \"\$(python ${RUN_PATH} -sc install=bash)\" " >> ~/.bashrc
