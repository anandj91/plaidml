import numpy as np
import sys
import plaidml
import plaidml.op as op
import plaidml.tile as ptile

# Generate input data
a_dim = (1, 26)
b_dim = (26, 1)
c_dim = (10, 20)

A = (np.ones(a_dim) + 1.) / 52.
B = (np.ones(b_dim) + 1.) / 52.
C = (np.ones(c_dim) + 1.) / 52.
#A = (np.random.rand(*a_dim) + 1.) / 52.
#B = (np.random.rand(*b_dim) + 1.) / 52.
#C = (np.random.rand(*c_dim) + 1.) / 52.

#plaidml._internal_set_vlog(4)
# Process data using PlaidML
ctx = plaidml.Context()

with plaidml.open_first_device(ctx) as dev:
    new_dtype = plaidml.DType.CUSTOM
    dtype = plaidml.DType.FLOAT32
    a_shape = plaidml.Shape(ctx, dtype, *a_dim)
    a = plaidml.Tensor(dev, a_shape)
    with a.mmap_discard(ctx) as view:
        view.copy_from_ndarray(A)
        view.writeback()
    b_shape = plaidml.Shape(ctx, dtype, *b_dim)
    b = plaidml.Tensor(dev, b_shape)
    with b.mmap_discard(ctx) as view:
        view.copy_from_ndarray(B)
        view.writeback()
    c_shape = plaidml.Shape(ctx, dtype, *c_dim)
    c = plaidml.Tensor(dev, c_shape)
    with c.mmap_discard(ctx) as view:
        view.copy_from_ndarray(C)
        view.writeback()

    x = ptile.Value.from_dimensions(a_dim, dtype, name='X')
    w = ptile.Value.from_dimensions(b_dim, dtype, name='W')
    u = op.cast(x, new_dtype)
    v = op.cast(w, new_dtype)
    #u = x
    #v = w
    m = op.MatMul(u, v).sole_output()
    n = m * 10.0
    e = op.exp(n)
    g = op.gradients(e, v)[0]
    nv = v - (g * 0.1)
    #o = op.cast(nv, dtype)
    o = nv
    f = ptile.compose(ctx, dev, [("W", w), ("X", x)], [("O", o)])
    invoker = plaidml.Invoker(ctx, f)
    invoker.set_input("X", a)
    invoker.set_input("W", b)
    shape = invoker.get_output_shape('O')
    d = plaidml.Tensor(dev, shape)
    invoker.set_output("O", d)
    invoker.invoke()

    dims = tuple(x.size for x in shape.dimensions)
    R = np.ndarray(dims)
    with d.mmap_current() as view:
        view.copy_to_ndarray(R)
    print('R', R)
