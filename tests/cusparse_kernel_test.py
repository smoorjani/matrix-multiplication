import torch
import random
import custom_mm
import numpy as np

n = 10
n_vals = 10

custom_mm.init_cusparse()


def gen_coords(num_vals, dim):
    coords = set()
    while len(coords) < num_vals:
        x, y = 0, 0
        while (x, y) == (0, 0):
            x, y = random.randint(0, dim-1), random.randint(0, dim-1)

        coords.add((x, y))
    return coords


a_coords = gen_coords(n_vals, n)
b_coords = gen_coords(n_vals, n)


def sparsify(coords, dim):
    mat = torch.zeros(n, n, dtype=torch.double)
    for x in range(0, dim):
        for y in range(0, dim):
            if (x, y) in coords:
                mat[x][y] = random.random()
    return mat


a = sparsify(a_coords, n)
b = sparsify(b_coords, n)

c = torch.zeros((n, n), device=torch.device('cuda'))

print('a: ', a)
print('b: ', b)

exp = a@b
custom_mm.cusparse_mmul(a, b, c)
our = c

diff = torch.nonzero(torch.subtract(exp, our))

print('# nonzero in exp:', torch.nonzero(exp).shape)
print('# nonzero in ours:', torch.nonzero(our).shape)

for x, y in diff:
    print(exp[x][y], our[x][y])

print('expected: ', exp)
print('ours(sparse): ', our)

custom_mm.destroy_cusparse()
