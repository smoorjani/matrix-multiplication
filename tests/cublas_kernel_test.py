import torch
import matmuls
import time
import sys

def test_result(function, a: torch.Tensor, b: torch.Tensor, kernel='both', transa=False, transb=False):
    output, expected = None, None
    tf, pt_tf = None, None

    if 'ours' in kernel or 'both' in kernel:
        t0 = time.perf_counter()
        output = function(a, b)
        tf = time.perf_counter() - t0
        output = output.cpu()
        print(f'Our time: {tf}')

    if 'pytorch' in kernel or 'both' in kernel:
        t0 = time.perf_counter()
        _a = a if not transa else a.transpose(-1, -2)
        _b = b if not transb else b.transpose(-1, -2)
        expected = torch.matmul(_a, _b).cpu()
        pt_tf = time.perf_counter() - t0
        print(f'PyTorch time: {pt_tf}')
         
    if 'both' in kernel: 
        try:
            assert (expected.shape == output.shape)
            assert (torch.allclose(expected, output))
        except AssertionError:
            # debugging statements if incorrect result
            print(torch.count_nonzero(output), torch.count_nonzero(expected))
            print(expected, output)
            raise AssertionError
    
    return tf if 'both' in kernel or 'ours' in kernel else pt_tf


def test_matmuls(a_dim, b_dim, transa=False, transb=False, kernel="both"):
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

    tf = test_result(operation, a, b, kernel=kernel, transa=transa, transb=transb)
    print(a_dim, b_dim, " passed!")
    return tf

def get_average_time(a_dim, b_dim, transa=False, transb=False, kernel="both", iters=5):
    total_time = [test_matmuls(a_dim, b_dim, transa=transa, transb=transb, kernel=kernel) for _ in range(iters)]
    return sum(total_time)/iters

# specify the kernel argument with sys.argv[1]

# Small tests
#print(get_average_time((2, 4, 2), (2, 3, 2), iters=1, transb=True))
#print(get_average_time((2, 2, 4, 2), (2, 2, 3, 2), iters=1, transb=True))

# BERT Tests
# from transformers import BertForQuestionAnswering
# model = BertForQuestionAnswering.from_pretrained('bert-large-uncased').cuda()
print(get_average_time((256, 16, 512, 512), (256, 16, 512, 64), iters=3))
print(get_average_time((16, 16, 512, 64), (16, 16, 512, 64), iters=3, transb=True))

# Large Tests
#print(get_average_time((64, 4096, 4096), (64, 4096, 4096), 2))
