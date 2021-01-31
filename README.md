# SpMM in PyTorch
Sparse Matrix Multiplication Kernels incorporated into Pytorch.

## Setup instructions

All necessary steps are covered in the `setup.sh` script.
Run the script using `sh setup.sh`.

## Binding with PyTorch
The following steps tell you how to take a CUDA kernel and bind it to PyTorch.

### Writing the C++ Wrapper
*Refer to `src/custom_mm.cpp` for an example with cuBLAS/cuSPARSE kernels implemented.*

You can write separate helpers to initialize the handles and use a global variable to store them. This variable persists after you import your module.

Note that torch passes in tensors by column major which may require some transposing. You can get a raw data pointer to the tensor by calling `tensor.data_ptr<dtype>()`.

When finished implementing the C++ wrapper, make sure to include it in the Pybind macro for the Python interpreter. The first argument is the method name, the second is the address of the C++ method, and the third is the description.

### Building the Python Module
Reconfigure the `setup.py` to point to your `.cpp` API and rename the module to whatever you want.
Run `python setup.py install` to install your module.

### Using the Module With PyTorch
*Refer to `tests/matmuls.py` for an examples.*

You will need to write a PyTorch InplaceFunction. For this you need to define a forward and backward pass and you will need to implement the gradient for the backward pass. You can simple copy the example for this and replace whatever `matmul` with the name of your function defined in the Pybind entrypoint.

It may help to define a wrapper as well for your matrix multiplication to deal with batched matrix multiplications or any error handling. Examples are also in the file.

Once you write the InplaceFunction, all you need to call it is `inplaceFunctionMatMul.apply(A,B)` where `inplaceFunctionMatMul` is the name of your extended InplaceFunction and `A` and `B` are your tensors.




