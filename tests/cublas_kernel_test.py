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
        _a = a if not transa else a.transpose(-1, -2)
        _b = b if not transb else b.transpose(-1, -2)
        expected = torch.matmul(_a, _b).cpu()
        pt_tf = time.time() - t0
        print(f'PyTorch time: {pt_tf}')
    '''     
    # debugging
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
        operation = matmuls.cublasTransabMM.apply
    elif transa:
        operation = matmuls.cublasTransaMM.apply
    elif transb:
        operation = matmuls.cublasTransbMM.apply
    else:
        operation = matmuls.cublasMM.apply
    tf, result = test_result(operation, a, b, transa=transa, transb=transb)
    assert result
    print(a_dim, b_dim, " passed!")
    return tf

def get_average_time(a_dim, b_dim, transa=False, transb=False, iters=5):
    total_time = 0
    for i in range(iters):
        total_time += test_matmuls(a_dim, b_dim, transa=transa, transb=transb)
    return total_time/iters

# BERT Tests

print(get_average_time((512, 512), (512, 64), iters=1))
print(get_average_time((2, 4), (2, 3), iters=1, transb=True))

# print(get_average_time((256, 16, 512, 512), (256, 16, 512, 64), iters=1))
# print(get_average_time((2, 2, 2, 4), (2, 2, 2, 3), iters=1, transb=True))
#print(get_average_time((16, 16, 512, 64), (16, 16, 512, 64), iters=1, transb=True))

# Large Tests
#print(get_average_time((64, 4096, 4096), (64, 4096, 4096), 2))

custom_mm.destroy_cublas()
