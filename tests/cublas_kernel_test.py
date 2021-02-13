import torch
import custom_mm
import matmuls

custom_mm.init_cublas()


def test_result(function, a: torch.Tensor, b: torch.Tensor):
    expected = torch.matmul(a, b)
    output = function(a, b)
    assert(expected.shape == output.shape)
    assert(torch.allclose(expected, output))
    return True


def test_raw_cublas_matmul(a_dim, b_dim):
    a = torch.rand(a_dim)
    b = torch.rand(b_dim)
    assert test_result(custom_mm.cublas_mmul, a, b)


def test_matmuls(a_dim, b_dim):
    a = torch.rand(a_dim)
    b = torch.rand(b_dim)
    assert test_result(matmuls.cublasMM.apply, a, b)


test_raw_cublas_matmul((8, 64), (64, 8))
test_raw_cublas_matmul((8, 64, 16), (16, 8))
test_raw_cublas_matmul((8, 64, 16), (8, 16, 8))
test_raw_cublas_matmul((1, 8, 64, 16), (1, 8, 16, 8))
test_raw_cublas_matmul((2, 8, 64, 16), (2, 8, 16, 8))

test_matmuls((8, 64), (64, 8))
test_matmuls((8, 64, 16), (16, 8))
test_matmuls((8, 64, 16), (8, 16, 8))
test_matmuls((1, 8, 64, 16), (1, 8, 16, 8))
test_matmuls((2, 8, 64, 16), (2, 8, 16, 8))

custom_mm.destroy_cublas()
