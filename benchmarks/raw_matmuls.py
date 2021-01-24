import torch_blocksparse
from custom_mm import cublas_mmul, cusparse_mmul
import time
import numpy as np
import torch


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
        return None


def test_kernel(matmul, a, b):
    t_init = time.time()
    assert a.shape[0] == b.shape[0]
    assert len(a.shape) == len(b.shape) == 3

    c = torch.stack([matmul(a[i], b[i]) for i in range(a.shape[0])]).cuda()
    t_final = time.time() - t_init

    print('Execution time for {num_samples} multiplications: {time}'.format(
        num_samples=a.shape[0], time=t_final))
    return c


num_samples = 1000
dim = 1024

a, b = generate_dataset(num_samples=num_samples, dim=dim, seed=0)
test_kernel(torch.matmul, a, b)
test_kernel(cublas_mmul, a, b)
test_kernel(cusparse_mmul, a, b)

H, M, N, K = num_samples, dim, dim, dim
block = 16
layout = torch.randint(0, 2, (H, M//block, N//block))
blocksparse_mmul = torch_blocksparse.MatMul(
    layout, block, 'sdd', trans_a=True, trans_b=False)

test_kernel(blocksparse_mmul, a, b)
