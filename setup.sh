#!/bin/bash

apt-get install python3.7 -y
apt-get install python3-pip -y 
apt-get install python3-dev -y

cd ..
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch/
pip3 install cython
pip3 install -r requirements.txt

mkdir build_libtorch && cd build_libtorch
export TORCH_CUDA_ARCH_LIST="7.0"
python3 ../tools/build_libtorch.py

cd /home/user/src
