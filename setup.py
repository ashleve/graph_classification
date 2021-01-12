#!/usr/bin/env python
from setuptools import setup, find_packages


setup(
    name='graph_classification',
    version='0.0.0',
    description='Graph classification tasks',
    author='hobogalaxy',
    author_email='',
    url='https://github.com/hobogalaxy/graph-classification',
    install_requires=['pytorch-lightning', 'hydra-core'],
    packages=find_packages(),
)
