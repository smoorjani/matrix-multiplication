import torch
import custom_mm
import matmuls

custom_mm.init_cublas()


def test_result(function, a: torch.Tensor, b: torch.Tensor):
    expected = torch.matmul(a, b)
    output = function(a, b).cpu()
   
    ''' 
    print('A: ', a)
    print('B: ', b)
    print('Expected: ', expected)
    print('Output: ', output)
   
    if len(a.size()) == 3 and len(b.size()) == 3:

        if (a.size()[1] == b.size()[2]):
            print('A.T @ B.T: ', a.transpose(1,2) @ b.transpose(1,2))
            print('B @ A: ', b @ a)
        print('B.T @ A.T: ', b.transpose(1,2) @ a.transpose(1,2))
    '''
    assert(expected.shape == output.shape)
    assert(torch.allclose(expected, output))
    return True


def test_raw_cublas_matmul(a_dim, b_dim):
    a = torch.rand(a_dim)
    b = torch.rand(b_dim)
    assert test_result(custom_mm.cublas_mmul, a, b)
    print(a_dim, b_dim, " passed!")

def test_matmuls(a_dim, b_dim):
    a = torch.rand(a_dim)
    b = torch.rand(b_dim)
    assert test_result(matmuls.cublasMM.apply, a, b)
    print(a_dim, b_dim, " passed!")

#test_raw_cublas_matmul((8, 64), (64, 8))
#test_raw_cublas_matmul((8, 64, 16), (16, 8))
#test_raw_cublas_matmul((8, 64, 16), (8, 16, 8))
#test_raw_cublas_matmul((1, 8, 64, 16), (1, 8, 16, 8))
#test_raw_cublas_matmul((2, 8, 64, 16), (2, 8, 16, 8))


test_matmuls((8, 64), (64, 8))
test_matmuls((8, 64, 16), (16, 8))
test_matmuls((2, 3, 2), (2, 2, 4))
test_matmuls((8, 64, 16), (8, 16, 8))
test_matmuls((1, 8, 64, 16), (1, 8, 16, 8))
test_matmuls((2, 8, 64, 16), (2, 8, 16, 8))
test_matmuls((1, 16, 512, 64), (1, 16, 64, 512))

custom_mm.destroy_cublas()
