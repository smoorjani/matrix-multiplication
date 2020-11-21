# matrix-multiplication
Sparse mm operations done through cuBLAS and incorporated into Python.

## Setup instructions

All instructions are covered in the `setup.sh` script.
Run the script using `sh setup.sh`.

### Directory Structure
cublas_mm/
  build/
    CMakeFiles
    - other build files
  CMakeLists.txt
  cublas_mm.cpp
  cublas_mm_kernel.cu
  jit.py
  setup.py


### Build Instructions
`mkdir build`
`cd build`
`cmake -DCMAKE_PREFIX_PATH=$PWD/../../../libtorch ..`
`cmake --build . --config Release`


### Binding with PyTorch
You must remove the `.cu` file import from your `.cpp` file
(i.e. remove line 9 from `cublas_mm.cpp`)

Make sure you have `Ninja` and `setuptools` installed via pip.

Run the following with setuptools
`python3 setup.py install`

Or with JIT
`python3 jit.py`