#!/bin/bash

apt update & apt upgrade -y
apt install vim nano git cmake wget -y

mkdir -p /home/user/src
sudo chmod -R 777 /home/user/*
cd home/user/src

apt-get install python3.7 -y
apt-get install python3-pip -y 
apt-get install python3-dev -y
export PATH="$PATH:/usr/local/cuda/bin"

git clone --recursive https://github.com/pytorch/pytorch
cd pytorch/
pip3 install cython
pip3 install -r requirements.txt

mkdir build_libtorch && cd build_libtorch
export TORCH_CUDA_ARCH_LIST="7.0"
python ../tools/build_libtorch.py

cd /home/user/src
