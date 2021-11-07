import torch
import custom_mm
import matmuls
import time

custom_mm.init_cublas()
custom_mm.init_cublaslt()

def test_result(function, a: torch.Tensor, b: torch.Tensor, kernel='both', transa=False, transb=False):
    output, expected = None, None
    tf, pt_tf = None, None
    kernel = 'both'
    if 'ours' in kernel or 'both' in kernel:
        t0 = time.time()
        output = function(a, b)
        tf = time.time() - t0
        print(output.shape)
        print('CUDA after returning: ', output.is_cuda)
        output = output.cpu()
        print(f'Our time: {tf}')
    if 'pytorch' in kernel or 'both' in kernel:
        t0 = time.time()
        _a = a if not transa else a.t()
        _b = b if not transb else b.t()
        expected = torch.matmul().cpu(_a, _b)
        pt_tf = time.time() - t0
        print(f'PyTorch time: {pt_tf}')
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
    if 'both' in kernel: 
        try:
       	    assert (expected.shape == output.shape)
            assert (torch.allclose(expected, output))
        except AssertionError:
            print(torch.count_nonzero(output), torch.count_nonzero(expected))
            print(expected, output)
            return tf, False
    
    return tf if 'both' in kernel or 'ours' in kernel else pt_tf, True


def test_matmuls(a_dim, b_dim, transa=False, transb=False):
    a = torch.rand(a_dim).to('cuda')
    b = torch.rand(b_dim).to('cuda')
    if transa and transb:
        tf, result = test_result(matmuls.cublasTransabMM.apply, a, b, transa=True, transb=True)
    elif transa:
        tf, result = test_result(matmuls.cublasTransaMM.apply, a, b, transa=True)
    elif transb:
        tf, result = test_result(matmuls.cublasTransbMM.apply, a, b, transb=True)
    else:
        tf, result = test_result(matmuls.cublasMM.apply, a, b)
    assert result
    print(a_dim, b_dim, " passed!")
    return tf

def get_average_time(a_dim, b_dim, transa=False, transb=False, iters=5):
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
#print(get_average_time((64, 4096, 4096), (64, 4096, 4096), 2))
#print(get_average_time((256, 16, 512, 512), (256, 16, 512, 64), 10))
print(get_average_time((16, 16, 512, 64), (16, 16, 512, 64), 10))

custom_mm.destroy_cublas()
