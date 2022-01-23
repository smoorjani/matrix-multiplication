import torch
import random
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


a_coords = gen_coords(n_vals, n, n)
b_coords = gen_coords(n_vals, n, n)


def sparsify(coords, dim1, dim2):
    mat = torch.zeros(dim1, dim2, device=torch.device('cuda'))
    for x in range(0, dim1):
        for y in range(0, dim2):
            if (x, y) in coords:
                mat[x][y] = random.random()
    return mat

shapes = [
          ((n, n), (n, n)),
          ((n, n), (n, int(n/2))),
          ((2*n, n), (n, int(n/2)))
        ]

for (a_rows, a_cols), (b_rows, b_cols) in shapes:
    a = sparsify(a_coords, a_rows, a_cols)
    #b = sparsify(b_coords, n)
    b = torch.rand((b_rows, b_cols), device=torch.device('cuda'))

    c = torch.zeros((a_rows, b_cols), device=torch.device('cuda'))

    #print('a: ', a)
    #print('b: ', b)

    exp = a@b
    _a = a.t()
    _b = b.t()
    custom_mm.cusparse_mmul(a,b,c)
    our = c
    print('expected: ', exp)
    print('ours(sparse): ', our)
    print('same? ', torch.allclose(exp, our))
'''
flag = False
for ela in [a, _a, b, _b]:
    if flag:
        break
    for elb in [a, _a, b, _b]:
        if torch.allclose(ela, elb) or torch.allclose(ela.t(), elb):
            continue
        custom_mm.cusparse_mmul(ela, elb, c)
        our = c

        if torch.allclose(our, exp):
            print(ela, elb)
            print('success')
            flag=True
            break
'''
'''
diff = torch.nonzero(torch.subtract(exp, our))

print('# nonzero in exp:', torch.nonzero(exp).shape)
print('# nonzero in ours:', torch.nonzero(our).shape)

for x, y in diff:
    print(exp[x][y], our[x][y])
'''

#print('a.t() @ b.t(): ', a.t() @ b.t())
#print('a.t() @ b: ', a.t() @ b)
#print('a @ b.t(): ', a @ b.t())
#print('b.t() @ a.t(): ', b.t() @ a.t())
#print('b.t() @ a: ', b.t() @ a)
#print('b @ a.t(): ', b @ a.t())

custom_mm.destroy_cusparse()
