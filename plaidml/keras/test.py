import numpy as np
import sys
import plaidml
import plaidml.op as op
import plaidml.tile as ptile

# Generate input data
a_dim = (10, 26)
b_dim = (26, 20)
c_dim = (10, 20)
d_dim = (10, 20)

A = (np.random.rand(*a_dim) + 1.) / 52.
B = (np.random.rand(*b_dim) + 1.) / 52.
C = (np.random.rand(*c_dim) + 1.) / 52.

plaidml._internal_set_vlog(4)
# Process data using PlaidML
ctx = plaidml.Context()

with plaidml.open_first_device(ctx) as dev:
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

    s = ptile.Value.from_dimensions(a_dim, dtype, name='S')
    t = ptile.Value.from_dimensions(b_dim, dtype, name='T')
    m = op.MatMul(s, t).sole_output()
    l = op.log(m)
    r = l * 2
    print("++++++++++++++++++++++++++++++++++")
    sys.stdout.flush()
    f = ptile.compose(ctx, dev, [("T", t), ("S", s)], [("R", r)])
    print("++++++++++++++++++++++++++++++++++")
    sys.stdout.flush()
    invoker = plaidml.Invoker(ctx, f)
    invoker.set_input("S", a)
    invoker.set_input("T", b)
    invoker.set_output("R", d)
    invoker.invoke()

    with d.mmap_current() as view:
        R = view[:]
