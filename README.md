# matrix-multiplication
Sparse mm operations done through cuBLAS and incorporated into Python.

## Setup instructions

All instructions are covered in the `setup.sh` script.

wget https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.7.0.zip
unzip libtorch-shared-with-deps-1.7.0.zip

===DIRECTORY STRUCTURE===
cublas_mm/
  CMakeLists.txt
  cublas_mm.cpp
libtorch/ (unzipped libtorch)


===BUILD INSTRUCTIONS===
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$PWD/../../libtorch ..
cmake --build . --config Release