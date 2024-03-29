import torch
import random
import matmuls
import custom_mm
import numpy as np

n = 10
n_vals = 10

custom_mm.init_cusparse()


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
          ((512, 1024), (1024, 256))
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
    #c = torch.zeros((a_rows, b_cols), device=torch.device('cuda'))

    exp = a@b
    print(a.shape, b.shape)
    a = a.to_sparse_csr()
    our = matmuls.cusparseMM.apply(a,b)

    if torch.allclose(exp, our):
        print(f'\nTest {i} passed!\n')
    else:
        print(f'\nTest {i} failed!\n')
        print('expected: ', exp)
        print('ours(sparse): ', our)
        print('# nonzero in exp:', torch.nonzero(exp).shape)
        print('# nonzero in ours:', torch.nonzero(our).shape)

custom_mm.destroy_cusparse()
