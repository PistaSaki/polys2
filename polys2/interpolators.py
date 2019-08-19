import numpy as np
#import itertools as itt
import tensorflow as tf
from pitf import nptf
from .poly import get_bin_indices

def get_bin_index(bins, t, name = "bin_index"):
    """Return (scalar) index of bin containing t.
    Args:
        `t` is scalar, 
        `bins` is ordered 1D tensor
    """
    return tf.reduce_max(
        tf.where( 
            condition = tf.less_equal(bins, t), 
        ),
        name = name,
    )
    

    
def CR_spline_coeffs_2(t0, t1, t2, t3, t, dif = False):
    """
    Returns the coefficients of the control points p0, p1, p2, p3 
    in the Catmull-Rom spline.
    t is between t1, t2
    """
    s = (t - t1) / (t2 - t1)
    
    if not dif:
        sss = [1, s, s**2, s**3]
    elif dif == 1:
        sss = np.array([0, 1, 2 * s, 3 * s**2]) / (t2 - t1)
    else:
        raise Exception("Unsuported dif = " + str(dif))
        
        
    coeffs = np.array(
        sss
    ).dot([
                [0, 1, 0, 0], 
                [1, 0, 0, 0], 
                [-2, -3, 3, -1], 
                [1, 2, -2, 1]
            ]
    ).dot(
            np.diag([(t2 - t1)/(t2-t0), 1, 1, (t2-t1)/(t3-t1)]) 
    ).dot(
            [
                [-1, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, -1, 0, 1]                
            ]
    )
    
    return list(coeffs)
    
    
def get_spline_coeffs_1d_simple_tf(controls, t, method = None, dif = False):
    if len(t.shape) == 0:
        bin_index = get_bin_index(controls, t)
    elif len(t.shape) == 1:
        bin_index = get_bin_indices(controls, t)
    else:
        raise ValueError(
            "t can be either a scalar or 1D tensor. ndim(t) = " + str(len(t.shape)) 
        )

    len_controls = tf.cast(tf.shape(controls)[-1], tf.int64)

    i1 = tf.minimum(tf.maximum(bin_index, 0), len_controls -2)
    i0 = tf.maximum(i1 - 1, 0)
    i2 = tf.minimum(i1 + 1, len_controls -1)
    i3 = tf.minimum(i2 + 1, len_controls -1)

    iii = [i0, i1, i2, i3]

    ttt = [tf.gather(controls, ii) for ii in iii]
    
    coeffs = CR_spline_coeffs_2(*ttt, t = t, dif = dif)
    
    return iii, coeffs
    
    
def get_spline_coeffs_simple_tf( params, x, method = None, derivative = None, ravel_multi_index = False):
    """
    multidimensional analogue of get_spline_coeffs_1d_simple.
    
    'params' is a list of len n of 1D tensors. 
    (I.e our spline is n-dimensional and params contains for each coordinate the list of
    corresponding controls of our grid.)
    'x' is a 1D tensor of len n --- it is the point where we want to evalueate our spline.
    (`x` can be 2D tensor, then the function is applied to all rows)
    'derivative' is a natural number --- the index of the coordinate by which we want to differentiate.
    if it is None, we don't differentiate.
    
    We return multi_index_list, coeff_list.
    multi_index_list is a list of lists of len n of natural numbers (if ravel_multi_index)
        or a list (if ravel_multi_index == False)
    coef_list is a list (of the same len as index_list) of real numbers.
    It means: to calculate value at x, take the grid-point at position multi_index_list[k] 
    with coefficient coef_list[k].
    
    """
    assert len(x.shape) in [1, 2]
    
    index_list = [0]
    multi_index_list = [[]]
    coeff_list = [1]
    for k, controls in enumerate(params):
        ind, coef = get_spline_coeffs_1d_simple_tf(
            controls, t = x[..., k], method = method, dif = (derivative == k))
        
        len_controls = tf.cast(tf.shape(controls)[0], tf.int64)
        index_list = [ il * len_controls + i for il in index_list for i in ind]
        multi_index_list = [ ml + [i] for ml in multi_index_list for i in ind]
        coeff_list = [ cl *  c  for cl in coeff_list for c in coef]
        
        
    if ravel_multi_index:
        return index_list, coeff_list
    else:
        return multi_index_list, coeff_list

        
def evaluate_interpolator_tf_one_x(params, values, x, raveled = True):
    assert raveled
    assert len(x.shape) in [1]
    
    indices, coeffs = get_spline_coeffs_simple_tf(params, x, ravel_multi_index=True)
    
    ret = sum([ 
        c * tf.gather(values, i) 
        for i, c in zip(indices, coeffs)
    ])   
    
    return ret

    
def evaluate_interpolator_tf(params, values, x, raveled = True, crop_x = True):
    assert raveled
    assert len(x.shape) in [1, 2]
    floatX = nptf.np_dtype(x)
    
    if crop_x:
        min_par = tf.cast(tf.stack([c[0] for c in params]), floatX)
        max_par = tf.cast(tf.stack([c[-1] for c in params]), floatX)
                
        if len(x.shape) == 2:
            min_par = min_par[None, :]
            max_par = max_par[None, :]
        x = tf.maximum(min_par, tf.minimum(x, max_par))
    
    ## see below why we fret so much abou these shapes
    codomain_shape = tf.shape( values)[1:]
    codomain_ndim = tf.size(codomain_shape)
    coeff_new_shape = tf.concat(
        [
            tf.constant([-1], dtype = tf.int32), 
            tf.ones([codomain_ndim], dtype = tf.int32)
        ],
        axis = 0
    )
    
    indices, coeffs = get_spline_coeffs_simple_tf(params, x, ravel_multi_index=True)
    ## now `indices` is an ordinary list of 1D Tensors 
    ## coeffs
    
    ret = sum([ 
        tf.reshape(c, coeff_new_shape) * tf.gather(values, i) 
        for i, c in zip(indices, coeffs)
    ])   
    
#    if len(x.shape) == 1:
#        return ret[0]
#    elif len(x.shape) == 2:
#        return ret
    
    if len(x.shape) == 1:
        ret = ret[0]
        ret.set_shape(values.shape[1:])
        return ret
    elif len(x.shape) == 2:
        ret.set_shape([x.shape[0]] + list(values.shape)[1:] )
        return ret
        
        
class InterpolatorEvaluator_tf:
    """This class does the same job as `evaluate_interpolator_tf`.
    
    However when hou have one large batch of `x` and you want to evaluate many
    interpolators at `x` then it is much more efficient to create one
    fixed instance of `InterpolatorEvaluator_tf` and then evaluate using that.
    
    Attributes:
        params: list of vectors
        x: tensor of shape `[n]` or `[batch_len, n]`
        indices: list of length 4**n of integer tensors of shape `()` or `[batch_len]`
        coeffs: list of length 4**n of float tensors of shape `()` or `[batch_len]`
        
        
    """
    def __init__(self, params, x, crop_x = True, name = "interpolator_evaluator"):
        """
        Args:
            params: is a list of len `n` of 1D tensors. Here `n` is the dimension 
                of the domain of our interpolators.
            x: a tensor of shape `[batch_len, n]`, or `[n]`. Point(s) where we evaluate.
            name: string to create tensorflow name_scope
        """
        assert len(x.shape) in [1, 2]
        self.name = name
        floatX = nptf.np_dtype(x)
    
        with tf.name_scope(self.name):
            if crop_x:
                min_par = tf.cast(tf.stack([c[0] for c in params]), floatX)
                max_par = tf.cast(tf.stack([c[-1] for c in params]), floatX)
                if len(x.shape) == 2:
                    min_par = min_par[None, :]
                    max_par = max_par[None, :]
                x = tf.maximum(min_par, tf.minimum(x, max_par))
        
            
            self.params = params
            self.x = x
            self.indices, self.coeffs = get_spline_coeffs_simple_tf(params, x, ravel_multi_index=True)
        
    def eval_itp(self, values):
        """
        Args:
            values: tensor containig values of our interpolator at the control points.
             It has shape `[n_control_pts] + codomain_shape` where 
             `n_control_pts = np.prod([len(c) for c in params])`
             
        Return:
            a tensor of shape `[batch_len] + codomain_shape` or of `codomain_shape`
            depending on whether `x` is a batch of points or just one point.
        """
        x = self.x
        
        codomain_shape = tf.shape( values)[1:]
        codomain_ndim = tf.size(codomain_shape)
        coeff_new_shape = tf.concat(
            [
                tf.constant([-1], dtype = tf.int32), 
                tf.ones([codomain_ndim], dtype = tf.int32)
            ],
            axis = 0
        )

        ret = tf.add_n([ 
            tf.reshape(c, coeff_new_shape) * tf.gather(values, i) 
            for i, c in zip(self.indices, self.coeffs)
        ])   

        if len(x.shape) == 1:
            ret = ret[0]
            ret.set_shape(values.shape[1:])
            return ret
        elif len(x.shape) == 2:
            ret.set_shape([x.shape[0]] + list(values.shape)[1:] )
            return ret
        
    def __call__(self, values):
        return self.eval_itp(values)
    
    
class Interpolator_tf:
    def __init__(self, params, values, raveled = False):
        self.params = params
        self.var_ndim = var_ndim = len(params)
        
        if raveled:
            self.raveled_values = values
        else:
            self.unraveled_values = values
            old_shape = tf.shape(values)
            
            new_shape = tf.concat(
                values = [
                    tf.reduce_prod(old_shape[:var_ndim], keepdims=True),
                    old_shape[var_ndim:]
                ], 
                axis = 0
            )
            self.raveled_values = tf.reshape(values, new_shape)
            
    def __call__(self, x):
        return evaluate_interpolator_tf(
            params = self.params,
            values = self.raveled_values,
            x = x,
            raveled = True
        )