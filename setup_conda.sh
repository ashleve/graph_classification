#!/bin/bash

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "conda could not be found"
    exit
fi

# This line is needed for enabling conda env activation
source ~/miniconda3/etc/profile.d/conda.sh

# Configure conda env
read -rp "Enter environment name: " env_name
read -rp "Enter python version (recommended '3.8') " python_version
read -rp "Enter cuda version (recommended '10.2', or 'none' for CPU only): " cuda_version
read -rp "Enter pytorch version (recommended '1.9'): " pytorch_version

# Create conda env
conda create -y -n "$env_name" python="$python_version"
conda activate "$env_name"

# Install pytorch
if [ "$cuda_version" == "none" ]; then
    conda install -y pytorch=$pytorch_version torchvision torchaudio cpuonly -c pytorch
else
    conda install -y pytorch=$pytorch_version torchvision torchaudio cudatoolkit=$cuda_version -c pytorch
fi

# Install pytorch geometric
conda install -y pytorch-geometric -c rusty1s -c conda-forge

# Install requirements
pip install -r requirements.txt

# Print message
echo ""
echo "To activate this environment, use:"
echo "conda activate $env_name"
