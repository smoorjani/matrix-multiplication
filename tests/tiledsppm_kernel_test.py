import torch
import random
import tiledspmm

from numpy import random
random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


n = 1024
n_vals = n * n

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

for i, ((a_rows, a_cols), (b_rows, b_cols)) in enumerate(shapes):
    # sparse dense matrix multiplication
    a_n_vals = (a_rows * a_cols) / n_vals # scales number of values to be (1/n_vals)% sparsity
    a_coords = gen_coords(a_n_vals, a_rows, a_cols)
    a = sparsify(a_coords, a_rows, a_cols)
    
    b = torch.rand((b_rows, b_cols), device=torch.device('cuda'))
    c = torch.zeros((a_rows, b_cols), device=torch.device('cuda'))

    exp = a@b
    
    a = a.to_sparse()
    vals = torch.Tensor.values(a).to('cpu')
    indices = torch.Tensor.indices(a).type(torch.IntTensor)
    rows = indices[0].to('cpu')
    cols = indices[1].to('cpu')
    nnz = len(vals)

    layer = "layer" + str(i)
    tiledspmm.tiledspmm_inspect_coo(a_rows, a_cols, b_cols, nnz, rows, cols, vals, layer)
    
    '''
    a = a.to_sparse_csr()
    vals = torch.Tensor.values(a).to('cpu')
    displ = torch.Tensor.crow_indices(a).type(torch.LongTensor).to('cpu')
    indices = torch.Tensor.col_indices(a).type(torch.LongTensor).to('cpu')

    tiledspmm.tiledspmm_inspect_csr(a_rows, a_cols, b_cols, displ, indices, vals)
    '''
    tiledspmm.tiledspmm_mm(b, c, layer)

    if torch.allclose(exp, c):
        print(f'\nTest {i} passed!\n')
    else:
        print(f'\nTest {i} failed!\n')
        print('expected: ', exp)
        print('ours(sparse): ', c)
        print('# nonzero in exp:', torch.nonzero(exp).shape)
        print('# nonzero in ours:', torch.nonzero(c).shape)

tiledspmm.tiledspmm_clean()
