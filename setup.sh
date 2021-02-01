#!/bin/bash

apt update
apt upgrade -y

apt install python3.7 python3-pip python3-dev -y
apt install libxml2-dev libxslt-dev python-dev -y
apt install pkg-config libhdf5-100 libhdf5-dev -y
apt install llvm-10 llvm-10-dev build-essential -y
apt install libblas-dev liblapack-dev -y

pip install setuptools ninja

# exit git repo directory
cd ..

python -c "import torch"                                                                                                          
if [ $? -eq 1 ]
then
    git clone --recursive https://github.com/pytorch/pytorch
    cd pytorch/
    pip3 install cython
    pip3 install -r requirements.txt

    # build libtorch and pytorch
    # mkdir build_libtorch && cd build_libtorch
    # export TORCH_CUDA_ARCH_LIST="7.0"
    # python3 ../tools/build_libtorch.py
    cd /home/user/src/pytorch
    python3 setup.py install
    cd ..
fi

python -c "import torchvision"                                                                                                          
if [ $? -eq 1 ]
then
    # build torchvision for MNIST data
    git clone --recursive https://github.com/pytorch/vision
    cd vision/
    python3 setup.py install
    cd ..
fi

# alias for python3
echo "alias python=python3" >> ~/.bashrc
source ~/.bashrc