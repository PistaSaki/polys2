"""
Basic functions for working with polynomials encoded into tensors.
Keeping things simple, no object-oriented structure.
"""
import numpy as np
import tensorflow as tf

from scipy.special import binom
import numpy.linalg as la


import tensor_utils as pit
import batch_utils as pib

#%%
        
##################################
## Auxiliary functions operating directly on np.arrays

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
        return x[..., None] ** np.arange(deg)
    
    assert x.shape[-1] == len(degs)
    
    # sometimes degs are tf.Dimension. Convert to ints:
    degs = [int(d) for d in degs]
    
    ret = 1
    for i in range(x.shape[-1]):
        xi = x[..., i]
        deg = degs[i]
        pows_xi = monomials_1var(xi, deg)
        selector = [None]*len(degs)
        selector[i] = slice(None)
        selector = [Ellipsis] + selector
        ret = ret * pows_xi[tuple(selector)]
        
    return ret



#%%
def eval_poly(coef, x, batch_ndim = None, var_ndim = None, val_ndim = None, ):
    """Return value at `x` of polynomial with coefficients `coef`."""
    def check_or_infer(n, n_infered, name):
        exception_string = (
                "Problem with shapes ({}): ".format(name) +
                " coef.shape=" + str(coef.shape) +
                " x.shape=" + str(x.shape) +
                " batch_ndim=" + str(batch_ndim) +
                " var_ndim="+ str(var_ndim) +
                " val_ndim=" + str(val_ndim)
            )
        
        assert n_infered >= 0, exception_string + "; n_infer < 0" 
        if n is None:
            return n_infered
        else:
            assert n == n_infered, (
                exception_string + "; given {} != infered {}".format(n, n_infered)
            )
            return n
        
#    if is_tf_object(x):
#        assert all(k is not None for k in [batch_ndim, var_ndim, val_ndim])
#    else:
    var_ndim = check_or_infer(var_ndim, int(x.shape[-1]), "var_ndim")
    batch_ndim = check_or_infer(batch_ndim, len(x.shape) - 1, "batch_ndim")
    val_ndim = check_or_infer(val_ndim, len(coef.shape) - batch_ndim - var_ndim, "val_ndim")

    degs = coef.shape[batch_ndim: batch_ndim + var_ndim]
    
    monoms = get_monomials(x, degs = degs)
    monoms = monoms[(Ellipsis, ) + (None, )*val_ndim]
    return tf.reduce_sum(
        coef * monoms, 
        axis = tuple(np.arange(batch_ndim, batch_ndim + var_ndim))
    )






#%%
def get_1D_Taylor_matrix(a, deg, trunc = None):
    """ Return matrix with shape `[trunc, deg]` af Taylor map at `a`.
    
    Taking (truncated) Taylor expansion at $a\in R$ defines a linear map $R[x] \to R[x]$.
    We return the matrix of this map.
    Our convention will be: $ deg(a_0 + a_1 x + ... + a_{n-1} x^{n-1}) = n $.
    """
    M = np.array([
        [binom(n, k) * a**(n-k) for k in range(n + 1)] + [0]*(deg - n - 1)
        for n in range(deg)
    ]).T
    
    if trunc is not None:
        M = M[:trunc]
        
    return M

#%%

def get_1d_Taylor_coef_grid(coef, poly_index, new_index, control_times, trunc = None):
    """
    Returns a tensor with one new index of length = len(control_times)
    """
    A = np.array([
        get_1D_Taylor_matrix(a, deg = coef.shape[poly_index], trunc = trunc).T
        for a in control_times        
    ], dtype = coef.dtype.as_numpy_dtype)
    
    if poly_index >= new_index:
        poly_index += 1
        

    ##It should work like this the following commented code and it does in numpy. 
    ##However tensorflow can't broadcast matmul yet, so it fails in tensorflow.
    #taylors = right_apply_map_along_batch(
    #    X = nptf.expand_dims(coef, new_index),
    #    A = A,
    #    batch_inds = [new_index],
    #    contract_inds = [poly_index],
    #    added_inds = [poly_index]
    #)

    ## Also this way could be written more concisely if tensorflow had equivalent of np.repeat:
    coef_repeated = tf.tile(
            tf.expand_dims(coef, new_index),
            reps = [len(control_times) if i == new_index else 1 for i in range(tf.ndim(coef) + 1)]
        )
    
    taylors = pit.right_apply_map_along_batch(
        X = coef_repeated,
        A = A,
        batch_inds = [new_index],
        contract_inds = [poly_index],
        added_inds = [poly_index]
    )
    return taylors



#%%

def get_1D_Taylors_to_spline_patch_matrix(a, b, deg):
    """
    `deg` is the degree of the Taylors. Thus the degree of the spline is 2 * deg.
    """
    Taylors_matrix = np.concatenate([
            get_1D_Taylor_matrix(a, deg = 2 * deg, trunc = deg),
            get_1D_Taylor_matrix(b, deg = 2 * deg, trunc = deg),
        ])
    
    #print(Taylors_matrix)
    return la.inv(Taylors_matrix)
                        
        
#get_1D_Taylors_to_spline_patch_matrix(0, 1, 2)
    

def get_Catmul_Rom_Taylors_1D(coef, control_index, control_times, added_index):
    assert len(control_times) == coef.shape[control_index]
         
    i0 = list(range(len(control_times)))
    c0 = coef
    
    # `t0` will be a tensor of the same ndim as `coef` but 
    # all dimenstions except at `control_index` are 1
    t_shape = np.ones(len(coef.shape), dtype=int)
    t_shape[control_index] = -1
    t0 = np.reshape(control_times, t_shape)
    
    i_minus = [0] + i0[:-1]
    i_plus = i0[1:] + [i0[-1]]
    
    t_minus = tf.gather(  t0, indices = i_minus, axis = control_index)
    c_minus = tf.gather(coef, indices = i_minus, axis = control_index)
    
    t_plus = tf.gather(  t0, indices = i_plus, axis = control_index)
    c_plus = tf.gather(coef, indices = i_plus, axis = control_index)
    
    der = (c_plus - c_minus) / (t_plus - t_minus)
    
    return tf.stack([c0, der], axis = added_index )
    
    
    
    

def array_poly_prod(a, b, batch_ndim = 0, var_ndim = None, truncation = None):
    """
    `a`, `b` are np.arrays
    truncation can be None, number or an array, 
    specifying the maximal allowed degrees  in the product.
    """
    ## take `a` to be the polynomial with values of lower ndim
    if a.ndim > b.ndim:
        a, b = b, a
        
    ## if `var_ndim` is missing, infer from shapes
    if var_ndim is None:
        var_ndim = a.ndim - batch_ndim
        
    ## check whether `a` has scalar values
    assert (a.ndim == batch_ndim + var_ndim) or (a.ndim == b.ndim), (
        "You have two possibilities: " + 
        "either the values of the two polynomials have the same ndim, " +
        "or one of them has scalar values."        
    )
    
    ## shape of values of the resulting polynomial
    val_shape = b.shape[batch_ndim + var_ndim:]
    
    ## add some dimensions at the end of `a` so that 
    ## it has the same ndim as `b`
    a = a[(Ellipsis, ) + (None, ) * (b.ndim - a.ndim)]
    
    ## degrees of the polys
    degs_a, degs_b = [np.array(x.shape)[batch_ndim: batch_ndim + var_ndim] for x in [a,b] ]
    deg_c = degs_a + degs_b - 1
    
    if truncation is not None:
        truncation = truncation * np.ones_like(deg_c)
        deg_c = np.array([min(t, d) for t, d in zip(truncation, deg_c)])
        
    c_batch_shape = pib.get_common_broadcasted_shape([
            a.shape[:batch_ndim], b.shape[:batch_ndim]
    ])
    
    c = np.zeros( c_batch_shape + list(deg_c) + list(val_shape) , dtype = np.promote_types(a.dtype, b.dtype))
    
    
    for i in range(np.prod(degs_a)):
        mi = np.unravel_index(i, degs_a)

        for j in range(np.prod(degs_b)):
            mj = np.unravel_index(j, degs_b)
            
            #print(mi, mj, deg_c)

            if all(np.add(mi, mj) < deg_c):
                batches = (slice(None),)*batch_ndim
                a_index = batches + mi
                b_index = batches + mj
                c_index = batches + tuple(np.add(mi, mj))
                c[c_index] += a[a_index] * b[b_index]
 
    return c

    

def get_spline_from_taylors_1D_OLD(taylor_grid_coeffs, bin_axis, polynom_axis, control_times):
    par = control_times
    taylor_grid = taylor_grid_coeffs
    
    # in the first step we put into polynom_axis the taylor_polynomials of the two consecutive bins
    start_selector = [slice(None)] * len(taylor_grid.shape)
    start_selector[bin_axis] = slice(None, taylor_grid.shape[bin_axis] - 1)
    start_selector = tuple(start_selector)

    end_selector = [slice(None)] * len(taylor_grid.shape)
    end_selector[bin_axis] = slice(1, None)
    end_selector = tuple(end_selector)
    
    stacked_taylors = tf.concat(
        [
            taylor_grid[start_selector], 
            taylor_grid[end_selector],
        ],
        axis = polynom_axis
    )
    
    
    ## now apply in each bin the corresponding "spline trasformation" i.e inverse of taking taylors 
    A = np.array(
        [
            get_1D_Taylors_to_spline_patch_matrix(
                    0, 1, b, deg = int(taylor_grid.shape[polynom_axis])
                ).T
            for a, b in zip(par[:-1], par[1:])
        ],
        dtype = tf.np_dtype(taylor_grid_coeffs)
    )

    coef = pit.right_apply_map_along_batch(
        X = stacked_taylors, A = A, 
        batch_inds = [bin_axis], contract_inds = [polynom_axis], added_inds = [polynom_axis]
    )
    
    
    return coef
    
def get_spline_from_taylors_1D(taylor_grid_coeffs, bin_axis, polynom_axis, control_times):
    par = control_times
    taylor_grid = taylor_grid_coeffs
    
    # in the first step we put into polynom_axis the taylor_polynomials of the two consecutive bins
    start_selector = [slice(None)] * len(taylor_grid.shape)
    start_selector[bin_axis] = slice(None, taylor_grid.shape[bin_axis] - 1)
    start_selector = tuple(start_selector)

    end_selector = [slice(None)] * len(taylor_grid.shape)
    end_selector[bin_axis] = slice(1, None)
    end_selector = tuple(end_selector)
    
    stacked_taylors = tf.concat(
        [
            taylor_grid[start_selector], 
            taylor_grid[end_selector],
        ],
        axis = polynom_axis
    )
    
    dtype = tf.np_dtype(taylor_grid_coeffs)
    deg = int(taylor_grid.shape[polynom_axis])
    ## reparametrization matrices for expressing the taylors in bin-scaled 
    ## Note that we have already stuck the two taylors together along the polynom-axis
    ## so our diagonal reparametrization matrices have the diagonal repeated twice
    ## ( thus the resulting matrix has dimension 2 * deg)
    RM = np.array(
        [
            np.diag([ (b- a)**k for k in range(deg) ] * 2)
            for a, b in zip(par[:-1], par[1:])
        ],
        dtype = dtype
    )
    
    ## now apply in each bin the corresponding "spline trasformation" i.e inverse of taking taylors 
    SM = np.array(
        [
            get_1D_Taylors_to_spline_patch_matrix(
                    0, 1, deg = deg
                ).T
            for a, b in zip(par[:-1], par[1:])
        ],
        dtype = dtype
    )

    coef = pit.right_apply_map_along_batch(
        X = stacked_taylors, A = RM @ SM, 
        batch_inds = [bin_axis], contract_inds = [polynom_axis], added_inds = [polynom_axis]
    )
    
    
    return coef


def get_1D_integral_of_piecewise_poly(                             
        coef, bin_axis, polynom_axis, control_times,
        polys_are_in_bin_coords = True
    ):
    deg = int(coef.shape[polynom_axis])
    def integration_functional(a, b, deg):
        n = np.arange(deg)
        if polys_are_in_bin_coords:  
            return 1/(n + 1) * (b - a)#**(n+1)
        else:
            return 1/(n + 1) * (b**(n+1) - a**(n+1))
    
    IM = np.array([
            integration_functional(a, b, deg)
            for a, b in zip(control_times[:-1], control_times[1:])
        ], dtype = tf.np_dtype(coef))
    
    # the following could be done with tensordot:
    return pit.right_apply_map_along_batch(
        X = coef, A = IM, 
        batch_inds = [], contract_inds = [bin_axis, polynom_axis], added_inds = []
    ) 
    
#get_1D_integral_of_piecewise_poly(
#    coef = np.array([
#        [0, 1, 0],
#        [1, 0, 0],
#    ]),
#    bin_axis = 0,
#    polynom_axis = 1,
#    control_times = [-1, 0, 1]
#)
    
def get_integral_of_spline_from_taylors_1D(
        taylor_grid_coeffs, bin_axis, polynom_axis, control_times,
        polys_are_in_bin_coords = True
    ):
    """
    This is basically a composition of 
    `get_spline_from_taylors_1D` and `get_1D_integral_of_piecewise_poly`
    I just believe that doing it in one step can be 2**n times faster
    where n is the number of dimensions, which is not too much.
    """
    par = control_times
    taylor_grid = taylor_grid_coeffs
    
    # in the first step we put into polynom_axis the taylor_polynomials of the two consecutive bins
    start_selector = [slice(None)] * len(taylor_grid.shape)
    start_selector[bin_axis] = slice(None, taylor_grid.shape[bin_axis] - 1)
    start_selector = tuple(start_selector)

    end_selector = [slice(None)] * len(taylor_grid.shape)
    end_selector[bin_axis] = slice(1, None)
    end_selector = tuple(end_selector)
    
    stacked_taylors = tf.concat(
        [
            taylor_grid[start_selector], 
            taylor_grid[end_selector],
        ],
        axis = polynom_axis
    )
    
    dtype = tf.np_dtype(taylor_grid_coeffs)
    deg = int(taylor_grid.shape[polynom_axis])
    ## reparametrization matrices for expressing the taylors in bin-scaled 
    ## Note that we have already stuck the two taylors together along the polynom-axis
    ## so our diagonal reparametrization matrices have the diagonal repeated twice
    ## ( thus the resulting matrix has dimension 2 * deg)
    RM = np.array(
        [
            np.diag([ (b- a)**k for k in range(deg) ] * 2)
            for a, b in zip(par[:-1], par[1:])
        ],
        dtype = dtype
    )
    
    ## now apply in each bin the corresponding "spline trasformation" i.e inverse of taking taylors 
    SM = np.array(
        [
            get_1D_Taylors_to_spline_patch_matrix(
                    0, 1, deg = deg
                ).T
            for a, b in zip(par[:-1], par[1:])
        ],
        dtype = dtype
    )
            
    

    ## finally we need to integrate 
    def integration_functional(a, b, deg):
        n = np.arange(deg)
        if polys_are_in_bin_coords:  
            return 1/(n + 1) * (b - a)#**(n+1)
        else:
            return 1/(n + 1) * (b**(n+1) - a**(n+1))
    
    IM = np.array([
            integration_functional(a, b, 2 * deg)
            for a, b in zip(control_times[:-1], control_times[1:])
        ], dtype = dtype)
    
    
    ## we multiply all the transformations 
    ## (we must make IM that is a batch of functionals into matrices)
    A = (RM @ SM @ IM[..., None])[..., 0]
    
    # the following could be done with tensordot:
    return pit.right_apply_map_along_batch(
        X = stacked_taylors, A = A, 
        batch_inds = [], contract_inds = [bin_axis, polynom_axis], added_inds = []
    ) 
    


