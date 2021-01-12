import torch
import random
import custom_mm
import numpy as np

a = torch.randn(5,4)
b = torch.randn(4,3)

print('expected: ', a@b)
print('ours: ', custom_mm.cublas_mmul(a,b))


n = 100
n_vals = 10
'''
a_vals = torch.randn(n_vals)
a_rowind = torch.randint(high=n, size=(n_vals,))
a_colind = torch.randint(high=n, size=(n_vals,))


b_vals = torch.randn(n_vals)
b_rowind = torch.randint(high=n, size=(n_vals,))
b_colind = torch.randint(high=n, size=(n_vals,))

_a = torch.sparse.FloatTensor(a_ind.unsqueeze(0), a_vals, (n,n))
a = _a.to_dense()
_b = torch.sparse.FloatTensor(b_ind.unsqueeze(0), b_vals, (n,n))
b = _b.to_dense()
'''

a = torch.zeros(n,n,dtype=torch.double)
b = torch.zeros(n,n,dtype=torch.double)

def gen_coords(num_vals, dim):
    coords = set()
    while len(coords) < num_vals:
        x, y = 0, 0
        while (x, y) == (0, 0):
            x, y = random.randint(0, dim-1), random.randint(0, dim-1)

        coords.add((x,y))
    return coords

a_coords = gen_coords(n_vals, n)
b_coords = gen_coords(n_vals, n)

def sparsify(mat, coords, dim):
    for x in range(0, dim):
        for y in range(0, dim):
            if (x,y) in coords:
                mat[x][y] = random.random()
    return mat

a = sparsify(a, a_coords, n)
b = sparsify(b, b_coords, n)

exp = a@b
our = custom_mm.cusparse_mmul(a,b)

print(torch.nonzero(torch.subtract(exp, our)))

print('expected: ', a@b)
print('ours(sparse): ', our) 
