import torch
import custom_mm
import matmuls
import time

custom_mm.init_cublas()
custom_mm.init_cublaslt()

def test_result(function, a: torch.Tensor, b: torch.Tensor):
    t0 = time.time()
    expected = torch.matmul(a, b).cpu()
    torch.cuda.synchronize()
    print(f'PyTorch time: {time.time() - t0}')
    t0 = time.time()
    output = function(a, b).cpu()
    tf = time.time() - t0
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
    
    try:
    	assert (expected.shape == output.shape)
    	assert (torch.allclose(expected, output))
    except AssertionError:
        print(torch.count_nonzero(output), torch.count_nonzero(expected))
        print(expected, output)
        return tf, False
   
    return 0, True


def test_matmuls(a_dim, b_dim):
    a = torch.rand(a_dim).to('cuda')
    b = torch.rand(b_dim).to('cuda')
    tf, result = test_result(matmuls.cublasMM.apply, a, b)
    assert result
    print(a_dim, b_dim, " passed!")
    return tf

def get_average_time(a_dim, b_dim, iters=5):
    total_time = 0
    for i in range(iters):
        total_time += test_matmuls(a_dim, b_dim)
    return total_time/iters

'''
test_matmuls((8, 64), (64, 8))
test_matmuls((8, 64, 16), (16, 8))
test_matmuls((2, 3, 2), (2, 2, 4))
test_matmuls((8, 64, 16), (8, 16, 8))
test_matmuls((64, 4096, 4096), (64, 4096, 4096))
test_matmuls((1, 8, 64, 16), (1, 8, 16, 8))
test_matmuls((2, 8, 64, 16), (2, 8, 16, 8))
'''
#test_matmuls((64, 16, 512, 64), (64, 16, 64, 512))
#test_matmuls((64, 16, 512, 512), (64, 16, 512, 64))
#test_matmuls((1, 16, 512, 64), (1, 16, 64, 512))
#print(get_average_time((2, 3, 2), (2, 2, 4)))
#print(get_average_time((8, 64, 16), (8, 16, 8)))
print(get_average_time((64, 4096, 4096), (64, 4096, 4096), 2))
custom_mm.destroy_cublas()
