import numpy as np
import sys
import plaidml
import plaidml.op as op
import plaidml.tile as ptile

# Generate input data
a_dim = (10, 20)
b_dim = (10, 20)
c_dim = (10, 20)

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

    s = ptile.Value.from_dimensions(a_dim, dtype, name='S')
    t = ptile.Value.from_dimensions(b_dim, dtype, name='T')
    c_s = op.cast(s, plaidml.DType.CUSTOM)
    c_t = op.cast(t, plaidml.DType.CUSTOM)
    c_r = c_s + c_t
    r = op.cast(c_r, dtype)
    f = ptile.compose(ctx, dev, [("T", t), ("S", s)], [("R", r)])
    invoker = plaidml.Invoker(ctx, f)
    invoker.set_input("S", a)
    invoker.set_input("T", b)
    invoker.set_output("R", c)
    invoker.invoke()

    with c.mmap_current() as view:
        R = view[:]
    print(R)
