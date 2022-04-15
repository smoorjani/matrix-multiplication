import torch
import random
import matmuls
import custom_mm

n = 10
n_vals = 10

def gen_coords(num_vals, dim1, dim2):
    coords = set()
    while len(coords) < num_vals:
        x, y = 0, 0
        while (x, y) == (0, 0):
            x, y = random.randint(0, dim1-1), random.randint(0, dim2-1)

        coords.add((x, y))
    return coords

def sparsify(coords, dim1, dim2):
    # return a sparse randomly generated tensor
    mat = torch.zeros(dim1, dim2, device=torch.device('cuda'))
    for x in range(0, dim1):
        for y in range(0, dim2):
            if (x, y) in coords:
                mat[x][y] = random.random()
    return mat

shapes = [
          ((n, n), (n, n)),
          ((n, 2*n), (2*n, n)),
          ((n, n), (n, int(n/2))),
          ((2*n, n), (n, int(n/2))),
        ]

# use these for sparse-sparse matrix multiplication
# b_coords = gen_coords(n_vals, b_rows, b_cols)
# b = sparsify(b_coords, n)

for i, ((a_rows, a_cols), b_shape) in enumerate(shapes):
    # sparse dense matrix multiplication
    a_n_vals = (a_rows * a_cols) / n_vals # scales number of values to be (1/n_vals)% sparsity
    a_coords = gen_coords(a_n_vals, a_rows, a_cols)
    a = sparsify(a_coords, a_rows, a_cols)
    
    b = torch.rand(b_shape, device=torch.device('cuda'))
    c = torch.zeros((a_rows, b_shape[-1]), device=torch.device('cuda'))

    exp = a@b
    print(a.shape, b.shape)
    _a = a.t()
    a = _a.to_sparse_csr()
    

    vals = torch.Tensor.values(a)
    cols = torch.Tensor.col_indices(a).type(torch.IntTensor)
    offsets = torch.Tensor.crow_indices(a).type(torch.IntTensor)
    print('inspecting!')
    custom_mm.tiledspmm_naive_inspect(vals, cols, offsets, a_rows, a_cols, b_shape[-1])
    print('multiplying!')
    b = b.t()
    c = custom_mm.tiledspmm_naive_mm(b, c)
    c = c.t()
    print((a.t() @ b.t()).t())
    print('cleaning up!')
    custom_mm.tiledspmm_naive_clean()

    if torch.allclose(exp, c):
        print(f'\nTest {i} passed!\n')
    else:
        print(f'\nTest {i} failed!\n')
        print('expected: ', exp)
        print('ours(sparse): ', c)
        print('# nonzero in exp:', torch.nonzero(exp).shape)
        print('# nonzero in ours:', torch.nonzero(c).shape)

