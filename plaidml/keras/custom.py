import numpy as np
import sys
import plaidml
import plaidml.op as op
import plaidml.tile as ptile

# Generate input data
a_dim = (1, 26)
b_dim = (26, 1)
c_dim = (10, 20)
d_dim = (1, 1)

A = (np.random.rand(*a_dim) + 1.) / 52.
B = (np.random.rand(*b_dim) + 1.) / 52.
C = (np.random.rand(*c_dim) + 1.) / 52.

plaidml._internal_set_vlog(4)
# Process data using PlaidML
ctx = plaidml.Context()

with plaidml.open_first_device(ctx) as dev:
    new_dtype = plaidml.DType.CUSTOM
    dtype = plaidml.DType.FLOAT32
    a_shape = plaidml.Shape(ctx, dtype, *a_dim)
    a = plaidml.Tensor(dev, a_shape)
    with a.mmap_discard(ctx) as view:
        view[:] = A.flatten()
        view.writeback()
    b_shape = plaidml.Shape(ctx, dtype, *b_dim)
    b = plaidml.Tensor(dev, b_shape)
    with b.mmap_discard(ctx) as view:
        view[:] = B.flatten()
        view.writeback()
    c_shape = plaidml.Shape(ctx, dtype, *c_dim)
    c = plaidml.Tensor(dev, c_shape)
    with c.mmap_discard(ctx) as view:
        view[:] = C.flatten()
        view.writeback()
    d_shape = plaidml.Shape(ctx, dtype, *d_dim)
    d = plaidml.Tensor(dev, d_shape)

    x = ptile.Value.from_dimensions(a_dim, dtype, name='X')
    w = ptile.Value.from_dimensions(b_dim, dtype, name='W')
    #u = op.cast(x, new_dtype)
    #v = op.cast(w, new_dtype)
    u = x
    v = w
    g = op.MatMul(u, v).sole_output()
    #l = op.exp(m)
    #o = op.cast(l, dtype)
    #o = l
    #g = op.gradients(o, m)[0]
    print("++++++++++++++++++++++++++++++++++")
    sys.stdout.flush()
    f = ptile.compose(ctx, dev, [("W", w), ("X", x)], [("G", g)])
    print("++++++++++++++++++++++++++++++++++")
    sys.stdout.flush()
    invoker = plaidml.Invoker(ctx, f)
    invoker.set_input("X", a)
    invoker.set_input("W", b)
    invoker.set_output("G", d)
    invoker.invoke()

    with d.mmap_current() as view:
        R = view[:]
    print('R', R)
