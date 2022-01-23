# SpMM in PyTorch
Sparse Matrix Multiplication Kernels incorporated into Pytorch.

## Setup instructions

Run `python setup.py install` in both the master and the `src` directories.
This will build the overhead logic and baseline multiplications (i.e. cuBLAS), respectively.

**Deprecated. Only use for Yellowzone**
All necessary steps are covered in the `install_deps.sh` script.
Run the script using `bash install_deps.sh`.

## How to use
*Refer to `tests/` for examples with both cuBLAS and cuSPARSE.*

```python
import torch
from matmuls import cublasMM
from custom_mm import init_cublas, destroy_cublas

# create cublas handle (one-time overhead)
init_cublas()

a = torch.rand(8, 64)
b = torch.rand(64, 8)

# cublasMM is a torch.InplaceFunction with a forward and backward pass
# You should use this when the gradient needs to be calculate
c = cublasMM.apply(a, b)

# destroy cublas handle
destroy_cublas()
```

## Binding with PyTorch
The following steps tell you how to take a CUDA kernel and bind it to PyTorch.

### Writing the C++ Wrapper
*Refer to `src/custom_mm.cpp` for an example with cuBLAS/cuSPARSE kernels implemented.*

You can write separate helpers to initialize the handles and use a global variable to store them. This variable persists after you import your module.

Note that torch passes in tensors by column major which may require some transposing. You can get a raw data pointer to the tensor by calling `tensor.data_ptr<dtype>()`. However, it is important to know that this operation does not guarantee a tensor stored in contiguous memory. In order to ensure it is stored contiguously, you may call `tensor.contiguous()`.

When finished implementing the C++ wrapper, make sure to include it in the Pybind macro for the Python interpreter. The first argument is the method name, the second is the address of the C++ method, and the third is the description.

### Building the Python Module
Reconfigure the `setup.py` to point to your `.cpp` API and rename the module to whatever you want.
Run `python setup.py install` to install your module.

### Using the Module With PyTorch
*Refer to `matmuls.py` for an examples.*

You will need to write a PyTorch InplaceFunction. For this you need to define a forward and backward pass and you will need to implement the gradient for the backward pass. You can simple copy the example for this and replace whatever `matmul` with the name of your function defined in the Pybind entrypoint.

It may help to define a wrapper as well for your matrix multiplication to deal with batched matrix multiplications or any error handling. Examples are also in the file.

Once you write the InplaceFunction, all you need to call it is `inplaceFunctionMatMul.apply(A,B)` where `inplaceFunctionMatMul` is the name of your extended InplaceFunction and `A` and `B` are your tensors.

## How to use with Huggingface Transformers
For a BERT model, you can replace the multiplications in the attention mechanism as follows:

```python
import matmuls

...

# store the tensors as contiguous in memory for the data_ptr<float>() method.
query_layer = query_layer.contiguous()
key_layer = key_layer.contiguous()

# original torch matrix multiplication
attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

# our matrix multiplication
attention_scores = matmuls.cublasTransbMM.apply(query_layer, key_layer)
```

Ensure that there is an existing cuBLAS handle. You do not need to recreate the handle as our module will find it.


