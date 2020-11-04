#include <torch/torch.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/*
 https://pytorch.org/tutorials/advanced/cpp_frontend.html

wget https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.7.0.zip
unzip libtorch-shared-with-deps-1.7.0.zip

 ===DIRECTORY STRUCTURE===
 mm/
   CMakeLists.txt
   mm.cpp
 libtorch/ (unzipped libtorch)


 ===BUILD INSTRUCTIONS===
 mkdir build
 cd build
 cmake -DCMAKE_PREFIX_PATH=$PWD/../../libtorch ..
 cmake --build . --config Release
*/

void print_matrix(const float *A, int A_rows, int A_cols);
void fill_rand(float *A, int A_rows, int A_cols);
void mmul(const float *A, const float *B, float *C, const int m, const int k, const int n);
void random_mmul();


void random_mmul_wrapper() {
    random_mmul();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mmul", &random_mmul_wrapper, "Random Matrix Multiplication");
}

