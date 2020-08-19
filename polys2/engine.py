"""
Basic functions for working with polynomials encoded into tensors.
Keeping things simple, no object-oriented structure.
"""
import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import backend as K
from tensorflow.keras.backend import ndim

from scipy.special import binom
from . import tensor_utils as pit


def get_monomials(x, degs):
    """ Return all monomials of `x` up to order `degs`.
    
    Args:
        x: tensor (numpy or tensorflow) of shape `batch_shape + [n]` containing 
            a batch of points in n-dimensional space
        degs: tuple of ints of length n
    Returns:
        tensor `mon` of shape `batch_shape + degs`. 
        When there are no bathces and if 
        `x =  [x1, x2, ... xn]` and `[a1, ... an] < degs` then
        `mon[a1, ...an] = x1**a1 * ... xn**an`.
    """

    def monomials_1var(x, deg):
        return x[..., None] ** tf.range(deg, dtype=x.dtype)

    assert x.shape[-1] == len(degs)

    ret = 1
    for i in range(x.shape[-1]):
        xi = x[..., i]
        deg = degs[i]
        pows_xi = monomials_1var(xi, deg)
        selector = [None] * len(degs)
        selector[i] = slice(None)
        selector = [Ellipsis] + selector
        ret = ret * pows_xi[tuple(selector)]

    return ret


def _as_numpy_dtype(dtype):
    try:
        return dtype.as_numpy_dtype()
    except AttributeError:
        return dtype


def find_common_dtype(array_types, scalar_types):
    return np.find_common_type(array_types=[_as_numpy_dtype(t) for t in array_types],
                               scalar_types=[_as_numpy_dtype(t) for t in scalar_types])


def eval_poly(coef, x, batch_ndim=None, var_ndim=None, val_ndim=None, ):
    """Return value at `x` of polynomial with coefficients `coef`."""
    if var_ndim is None:
        var_ndim = x.shape[-1]
    else:
        tf.assert_equal(var_ndim, x.shape[-1])

    if batch_ndim is None:
        batch_ndim = ndim(x) - 1

    if val_ndim is None:
        val_ndim = ndim(coef) - batch_ndim - var_ndim

    degs = tf.shape(coef)[batch_ndim: batch_ndim + var_ndim]

    monoms = get_monomials(x, degs=degs)
    monoms = monoms[(Ellipsis,) + (None,) * val_ndim]
    return tf.reduce_sum(
        coef * monoms,
        axis=tf.range(batch_ndim, batch_ndim + var_ndim)
    )


def get_1D_Taylor_matrix(a, deg: int, trunc: int = None):
    """ Return matrix with shape `[trunc, deg]` af Taylor map at `a`.
    
    Taking (truncated) Taylor expansion at $a in R$ defines a linear map $R[x] / (x^deg)$ to $R[x] / (x^trunc)$.
    We return the matrix of this map.
    Our convention will be: $ deg(a_0 + a_1 x + ... + a_{n-1} x^{n-1}) = n $.

    Args:
        a: scalar or batch of scalars
        deg: degree of input polynomial
        trunc: degree of Taylor expansion

    Returns:
        tensor of shape `a.shape + [trunc, deg]`
    """
    a = tf.convert_to_tensor(a, dtype_hint=K.floatx())
    deg = int(deg)
    zero = tf.zeros_like(a)
    columns = [
        tf.stack([binom(n, k) * a ** (n - k) for k in range(n + 1)] + [zero for _ in range(deg - n - 1)], axis=-1)
        for n in range(deg)]
    M = tf.stack(columns, axis=-1)

    if trunc is not None:
        M = M[..., :trunc, :]

    return M


def get_1d_Taylor_coef_grid(coef, poly_index, new_index, control_times, trunc=None):
    """
    Returns a tensor with one new index of length = len(control_times)
    """
    A = get_1D_Taylor_matrix(control_times, deg=coef.shape[poly_index], trunc=trunc)

    if poly_index >= new_index:
        poly_index += 1

    taylors = pit.right_apply_map_along_batch(
        X=tf.expand_dims(coef, new_index),
        A=tf.linalg.matrix_transpose(A),
        batch_inds=[new_index],
        contract_inds=[poly_index],
        added_inds=[poly_index]
    )
    return taylors


def get_1D_Taylors_to_spline_patch_matrix(a, b, deg):
    """
    `deg` is the degree of the Taylors. Thus the degree of the spline is 2 * deg.
    """
    Taylors_matrix = tf.concat([
        get_1D_Taylor_matrix(a, deg=2 * deg, trunc=deg),
        get_1D_Taylor_matrix(b, deg=2 * deg, trunc=deg),
    ], axis=-2)

    # print(Taylors_matrix)
    return tf.linalg.inv(Taylors_matrix)


def get_Catmul_Rom_Taylors_1D(coef, control_index, control_times, added_index):
    tf.assert_equal(tf.shape(control_times)[0], tf.shape(coef)[control_index])

    i0 = list(range(len(control_times)))
    c0 = coef

    # `t0` will be a tensor of the same ndim as `coef` but 
    # all dimenstions except at `control_index` are 1
    t_shape = np.ones(len(coef.shape), dtype=int)
    t_shape[control_index] = -1
    t0 = tf.reshape(control_times, t_shape)

    i_minus = [0] + i0[:-1]
    i_plus = i0[1:] + [i0[-1]]

    t_minus = tf.gather(t0, indices=i_minus, axis=control_index)
    c_minus = tf.gather(coef, indices=i_minus, axis=control_index)

    t_plus = tf.gather(t0, indices=i_plus, axis=control_index)
    c_plus = tf.gather(coef, indices=i_plus, axis=control_index)

    der = (c_plus - c_minus) / (t_plus - t_minus)

    return tf.stack([c0, der], axis=added_index)


def _tf_ravel_multi_index(multi_index, dims):
    strides = tf.math.cumprod(dims, axis=-1, reverse=True, exclusive=True)
    return tf.reduce_sum(multi_index * strides, axis=-1)


def _stack_tensor_array(a, axis=0):
    b = a.stack()
    nd = ndim(b)
    perm = tf.concat([tf.range(1, axis + 1), [0], tf.range(axis + 1, nd)], axis=0)
    return tf.transpose(b, perm)


def poly_prod(a, b, batch_ndim=0, var_ndim=None, truncation=None, dtype=None) -> Tensor:
    dtype = dtype or K.floatx()
    a = tf.cast(a, dtype)
    b = tf.cast(b, dtype)

    # take `a` to be the polynomial with values of lower ndim
    if ndim(a) > ndim(b):
        a, b = b, a

    # if `var_ndim` is missing, infer from shapes
    if var_ndim is None:
        var_ndim = ndim(a) - batch_ndim

    # check whether `a` has scalar values or
    tf.Assert((ndim(a) == batch_ndim + var_ndim) | (ndim(a) == ndim(b)), data=[ndim(a), ndim(b)])
    #           message=(
    #     "You have two possibilities: either the values of the two polynomials have the same ndim "
    #     "or one of them has scalar values.")
    # )

    # shape of values of the resulting polynomial
    val_shape = tf.shape(b)[batch_ndim + var_ndim:]

    # add some dimensions at the end of `a` so that it has the same ndim as `b`
    a = a[(Ellipsis,) + (None,) * (ndim(b) - ndim(a))]

    # degrees of the polys
    degs_a, degs_b = [np.array(x.shape[batch_ndim: batch_ndim + var_ndim], np.int) for x in [a, b]]
    degs_c = degs_a + degs_b - 1

    if truncation is not None:
        degs_c = np.minimum(degs_c, truncation)

    c_batch_shape = tf.broadcast_dynamic_shape(tf.shape(a)[:batch_ndim], tf.shape(b)[:batch_ndim])

    def flatten_polynomial_dimensions(x):
        s = tf.shape(x)
        ndim = tf.shape(s)[0]
        val_ndim = ndim - batch_ndim - var_ndim
        batch_shape = s[:batch_ndim]
        degs_shape = s[batch_ndim: batch_ndim + var_ndim]
        val_shape = s[batch_ndim + var_ndim:]
        flatt_degs_shape = tf.concat([batch_shape, tf.reduce_prod(degs_shape, keepdims=True), val_shape], axis=0)
        x = tf.reshape(x, flatt_degs_shape)
        perm = tf.concat([[batch_ndim], tf.range(batch_ndim), tf.range(batch_ndim + 1, batch_ndim + 1 + val_ndim)],
                         axis=0)
        return tf.transpose(x, perm=perm)

    a_flat = flatten_polynomial_dimensions(a)
    b_flat = flatten_polynomial_dimensions(b)
    c_array = tf.TensorArray(dtype=dtype, size=tf.reduce_prod(degs_c))
    element_shape = tf.concat([c_batch_shape, val_shape], axis=0)
    for i in range(np.prod(degs_c)):
        c_array = c_array.write(i, tf.zeros(element_shape, dtype))

    for i in range(np.prod(degs_a)):
        mi = np.array(np.unravel_index(i, degs_a))

        for j in range(np.prod(degs_b)):
            mj = np.array(np.unravel_index(j, degs_b))

            m = mi + mj
            if np.all(m < degs_c):
                n = _tf_ravel_multi_index(m, dims=degs_c)
                c_array = c_array.write(n, c_array.read(n) + a_flat[i] * b_flat[j])

    c_shape = tf.concat([c_batch_shape, degs_c, val_shape], axis=0)
    c = _stack_tensor_array(c_array, axis=batch_ndim)
    c = tf.reshape(c, c_shape)

    return c


def get_spline_from_taylors_1d(taylor_grid_coeffs, bin_axis, polynom_axis, control_times):
    par = control_times
    taylor_grid = taylor_grid_coeffs

    # in the first step we put into polynom_axis the taylor_polynomials of the two consecutive bins
    start_selector = [slice(None)] * len(taylor_grid.shape)
    start_selector[bin_axis] = slice(None, taylor_grid.shape[bin_axis] - 1)
    start_selector = tuple(start_selector)

    end_selector = [slice(None)] * len(taylor_grid.shape)
    end_selector[bin_axis] = slice(1, None)
    end_selector = tuple(end_selector)

    stacked_taylors = tf.concat([taylor_grid[start_selector], taylor_grid[end_selector]], axis=polynom_axis)

    n_bins = tf.shape(stacked_taylors)[bin_axis]
    deg = int(taylor_grid.shape[polynom_axis])  # `deg` is degree of input taylor-grid. Degree of spline is `2*deg`.

    # reparametrization matrices for expressing the taylors in bin-scaled coordinates
    # Note that we have already concatenated the two taylors together along the polynom-axis
    # so our diagonal reparametrization matrices have the diagonal repeated twice
    # ( thus the resulting matrix has dimension 2 * deg)
    # RM.shape = [n_bins, 2*deg, 2*deg]
    RM = tf.linalg.diag(tf.stack([(par[1:] - par[:-1]) ** k for k in range(deg)] * 2, axis=1))
    tf.assert_equal(tf.shape(RM), [n_bins, 2 * deg, 2 * deg])

    # now apply in each bin the corresponding "spline transformation" i.e inverse of taking taylors
    SM = tf.repeat(input=tf.transpose(get_1D_Taylors_to_spline_patch_matrix(0, 1, deg=deg))[None, :, :],
                   repeats=n_bins, axis=0)
    tf.assert_equal(tf.shape(SM), [n_bins, 2 * deg, 2 * deg])

    coef = pit.right_apply_map_along_batch(X=stacked_taylors, A=RM @ SM, batch_inds=[bin_axis],
                                           contract_inds=[polynom_axis], added_inds=[polynom_axis])
    return coef


def get_1d_integral_of_piecewise_poly(coef, bin_axis, polynom_axis, control_times):
    par = control_times
    n_bins = tf.shape(par)[0] - 1
    deg = int(coef.shape[polynom_axis])

    n = tf.range(deg, dtype=par.dtype)
    IM = 1 / (n + 1) * (par[1:] - par[:-1])[:, None]
    tf.assert_equal(tf.shape(IM), [n_bins, deg])

    # the following could be done with tensordot:
    return pit.right_apply_map_along_batch(
        X=coef, A=IM, batch_inds=[], contract_inds=[bin_axis, polynom_axis], added_inds=[])


def get_integral_of_spline_from_taylors_1d(
        taylor_grid_coeffs, bin_axis, polynom_axis, control_times,
):
    """
    This is basically a composition of 
    `get_spline_from_taylors_1D` and `get_1D_integral_of_piecewise_poly`
    I just believe that doing it in one step can be 2**n times faster
    where n is the number of dimensions, which is not too much.
    """
    par = control_times
    taylor_grid = tf.convert_to_tensor(taylor_grid_coeffs, dtype_hint=K.floatx())

    # in the first step we put into polynom_axis the taylor_polynomials of the two consecutive bins
    start_selector = [slice(None)] * len(taylor_grid.shape)
    start_selector[bin_axis] = slice(None, -1)
    start_selector = tuple(start_selector)

    end_selector = [slice(None)] * len(taylor_grid.shape)
    end_selector[bin_axis] = slice(1, None)
    end_selector = tuple(end_selector)

    stacked_taylors = tf.concat([taylor_grid[start_selector], taylor_grid[end_selector]], axis=polynom_axis)

    n_bins = tf.shape(stacked_taylors)[bin_axis]
    deg = int(taylor_grid.shape[polynom_axis])  # `deg` is degree of input taylor-grid. Degree of spline is `2*deg`.

    # reparametrization matrices for expressing the taylors in bin-scaled coordinates
    # Note that we have already concatenated the two taylors together along the polynom-axis
    # so our diagonal reparametrization matrices have the diagonal repeated twice
    # ( thus the resulting matrix has dimension 2 * deg)
    # RM.shape = [n_bins, 2*deg, 2*deg]
    RM = tf.linalg.diag(tf.stack([(par[1:] - par[:-1]) ** k for k in range(deg)] * 2, axis=1))
    tf.assert_equal(tf.shape(RM), [n_bins, 2 * deg, 2 * deg])

    # now apply in each bin the corresponding "spline transformation" i.e inverse of taking taylors
    SM = tf.repeat(input=tf.transpose(get_1D_Taylors_to_spline_patch_matrix(0, 1, deg=deg))[None, :, :],
                   repeats=n_bins, axis=0)
    tf.assert_equal(tf.shape(SM), [n_bins, 2 * deg, 2 * deg])

    # finally we need to integrate
    # `IM.shape == [n_bins, deg]`
    n = tf.range(2 * deg, dtype=par.dtype)
    IM = 1 / (n + 1) * (par[1:] - par[:-1])[:, None]
    tf.assert_equal(tf.shape(IM), [n_bins, 2 * deg])

    # we multiply all the transformations (we must make IM that is a batch of functionals into matrices)
    A = (RM @ SM @ IM[..., None])[..., 0]

    # the following could be done with tensordot:
    return pit.right_apply_map_along_batch(X=stacked_taylors, A=A, batch_inds=[],
                                           contract_inds=[bin_axis, polynom_axis], added_inds=[])
