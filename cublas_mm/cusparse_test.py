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

diff = torch.nonzero(torch.subtract(exp, our))

for x, y in diff:
    print(exp[x][y], our[x][y])

print('expected: ', a@b)
print('ours(sparse): ', our) 
