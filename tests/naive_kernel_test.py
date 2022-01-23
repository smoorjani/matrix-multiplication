import torch
import custom_mm
import matmuls
import time
import sys

#custom_mm.init_cublas()

def test_result(a: torch.Tensor, b: torch.Tensor, kernel='both', transa=False, transb=False):
    output, expected = None, None
    tf, pt_tf = None, None
    if 'ours' in kernel or 'both' in kernel:
        _a = a if not transa else a.transpose(-1, -2)
        _b = b if not transb else b.transpose(-1, -2)
        t0 = time.perf_counter()
        output = matmuls.naiveMM.apply(a, b)
        tf = time.perf_counter() - t0
        output = output.cpu()
        print(f'Our time: {tf}')
    if 'pytorch' in kernel or 'both' in kernel:
        _a = a if not transa else a.transpose(-1, -2)
        _b = b if not transb else b.transpose(-1, -2)
        t0 = time.perf_counter()
        expected = torch.matmul(_a, _b).cpu()
        pt_tf = time.perf_counter() - t0
        print(f'PyTorch time: {pt_tf}')
         
    if 'both' in kernel: 
        try:
       	    assert (expected.shape == output.shape)
            assert (torch.allclose(expected, output))
        except AssertionError:
            print(torch.count_nonzero(output), torch.count_nonzero(expected))
            print(expected, output)
            return tf, False
    
    return tf if 'both' in kernel or 'ours' in kernel else pt_tf, True


def test_matmuls(a_dim, b_dim, transa=False, transb=False, kernel="both"):
    a = torch.rand(a_dim).to('cuda')
    b = torch.rand(b_dim).to('cuda')

    tf, result = test_result(a, b, kernel=kernel, transa=transa, transb=transb)
    assert result
    print(a_dim, b_dim, " passed!")
    return tf

def get_average_time(a_dim, b_dim, transa=False, transb=False, kernel="both", iters=5):
    total_time = 0
    for i in range(iters):
        total_time += test_matmuls(a_dim, b_dim, transa=transa, transb=transb, kernel=kernel)
    return total_time/iters

# BERT Tests

print(get_average_time((512, 512), (512, 64), iters=1))
#print(get_average_time((256, 16, 512, 512), (256, 16, 512, 64), iters=3))
print(get_average_time((2, 4, 2), (2, 3, 2), iters=1, transb=True))
#print(get_average_time((2, 2, 4, 2), (2, 2, 3, 2), iters=1, transb=True))
#print(get_average_time((16, 16, 512, 64), (16, 16, 512, 64), iters=3, transb=True))

# Large Tests
#print(get_average_time((64, 4096, 4096), (64, 4096, 4096), 2))

#custom_mm.destroy_cublas()
