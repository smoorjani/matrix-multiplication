#!/bin/bash

wget https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.7.0.zip
unzip libtorch-shared-with-deps-1.7.0.zip

sudo apt-get install python3-dev

cd cublas_mm
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$PWD/../../libtorch ..
cmake --build . --config Release