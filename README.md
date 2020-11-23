# matrix-multiplication
Sparse mm operations done through cuBLAS and incorporated into Python.

## Setup instructions

All instructions are covered in the `setup.sh` script.
Run the script using `bash setup.sh`.

### Binding with PyTorch
Make sure you have `Ninja` and `setuptools` installed via pip.

Run the following with setuptools
`python3 setup.py install`

### CMake Build Instructions
`mkdir build`
`cd build`
`cmake -DCMAKE_PREFIX_PATH=$PWD/../../../libtorch ..`
`make -j`


