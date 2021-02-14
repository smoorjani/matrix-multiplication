import torch
from scipy.sparse import random
import numpy as np
import time
import logging
from custom_mm import (
    cublas_mmul,
    cusparse_mmul,
    init_cublas,
    destroy_cublas,
    init_cusparse,
    destroy_cusparse
)

init_cublas()
init_cusparse()

LOG = "./random_tensor_benchmark.log"
logging.basicConfig(filename=LOG, filemode="w", level=logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(logging.ERROR)
logging.getLogger("").addHandler(console)

logger = logging.getLogger(__name__)


def generate_dataset(num_samples: int = 1000, dim: int = 1024,
                     seed: int = None, sparsity: float = 0):

    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if not sparsity:
        a = torch.randn(num_samples, dim, dim)
        b = torch.randn(num_samples, dim, dim)

        return (a, b)
    else:
        # TODO: generate sparse matrices
        # - one idea is to generate
        # https://stackoverflow.com/questions/64553148/how-to-convert-a-pytorch-sparse-coo-tensor-into-a-pytorch-dense-tensor
        # - other idea is to delete
        a = torch.stack([torch.tensor(
            random(dim, dim, density=1-sparsity).toarray()) for _ in range(num_samples)])
        b = torch.stack([torch.tensor(
            random(dim, dim, density=1-sparsity).toarray()) for _ in range(num_samples)])
        return (a, b)


def test_kernel(matmul, a, b):
    t_init = time.time()
    assert a.shape[0] == b.shape[0]
    assert len(a.shape) == len(b.shape) == 3

    c = torch.stack([matmul(a[i], b[i]) for i in range(a.shape[0])])
    t_final = time.time() - t_init

    logger.debug('Execution time for {num_samples} multiplications: {time}\n'.format(
        num_samples=a.shape[0], time=t_final))
    logger.debug('Average time for one multiplication: {time}\n'.format(
        time=t_final/a.shape[0]))
    return c


num_samples = 1000

dims = [1024, 4096, 8192, 12288, 16384]
sparsity_levels = [0, 0.25, 0.5, 0.75, 0.9, 0.99]

for sparsity in sparsity_levels:
    for dim in dims:
        a, b = generate_dataset(num_samples=num_samples,
                                dim=dim, seed=0, sparsity=sparsity)
        logger.debug("Testing {} by {} matrices with {} percent sparsity.\n".format(
            dim, dim, sparsity*100))

        logger.debug("Regular Torch Matmul: \n")
        test_kernel(torch.matmul, a, b)

        logger.debug("cuBLAS Matmul: \n")
        test_kernel(cublas_mmul, a, b)

        logger.debug("cuSPARSE Matmul: \n")
        _a = a.type(torch.DoubleTensor)
        _b = b.type(torch.DoubleTensor)
        test_kernel(cusparse_mmul, _a, _b)

# # TODO: fix issue with "RuntimeError: operation does not have an identity."
# print("BlockSparse Matmul: \n")
# H, M, N, K = num_samples, dim, dim, dim
# block = 16
# layout = torch.randint(0, 2, (H, M//block, N//block))
# blocksparse_mmul = torch_blocksparse.MatMul(
#     layout, block, 'sdd', trans_a=True, trans_b=False)

# test_kernel(blocksparse_mmul, a, b)

destroy_cublas()
destroy_cusparse()