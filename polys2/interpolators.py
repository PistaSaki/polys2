import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import backend as K
from pitf import nptf
from typing import List, Tuple

from .poly import get_bin_indices    
    
def CR_spline_coeffs(t0, t1, t2, t3, t, dif = False, dtype = None) -> List[Tensor]:
    """
    Returns the coefficients of the control points p0, p1, p2, p3 
    in the Catmull-Rom spline.
    t is between t1, t2
    """
    dtype = dtype or K.floatx()
    t0, t1, t2, t3, t = [tf.cast(x, dtype) for x in [t0, t1, t2, t3, t]]
    assert t0.shape == t1.shape == t2.shape == t3.shape
    
    s = (t - t1) / (t2 - t1)
    
    assert dif in [0, 1]
    if dif:
        sss = tf.stack([tf.zeros_like(s), tf.ones_like(s), 2 * s, 3 * s**2], axis=-1) / (t2 - t1)
    else:
        sss = tf.stack([tf.ones_like(s), s, s**2, s**3], axis=-1)
        
    coeffs = sss 

    if len(sss.shape) == 1: # hack to make matmul work 
        coeffs = coeffs[None, :]
        
    
    
    coeffs = coeffs @ [[0, 1, 0, 0], 
                       [1, 0, 0, 0], 
                       [-2, -3, 3, -1], 
                       [1, 2, -2, 1]]
    coeffs = coeffs * tf.stack([(t2 - t1)/(t2-t0), tf.ones_like(t0), 
                                tf.ones_like(t0), (t2-t1)/(t3-t1)], axis=-1) 
    coeffs = coeffs @ [[-1, 0, 1, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, -1, 0, 1]]
    
    if len(sss.shape) == 1: # caused by hack to make matmul work 
        coeffs = coeffs[0]
        
    return tf.unstack(coeffs, axis=-1)
    
    
def get_spline_coeffs_1d(controls, t, method = None, dif = False, dtype=None
                         ) -> Tuple[List[Tensor], List[Tensor]]:
    
    dtype = dtype or K.floatx()
    controls, t = [tf.cast(x, dtype) for x in [controls, t]]
    
    bin_index = get_bin_indices(controls, t)
    
    len_controls = tf.cast(tf.shape(controls)[-1], tf.int32)

    i1 = tf.minimum(tf.maximum(bin_index, 0), len_controls -2)
    i0 = tf.maximum(i1 - 1, 0)
    i2 = tf.minimum(i1 + 1, len_controls -1)
    i3 = tf.minimum(i2 + 1, len_controls -1)

    iii = [i0, i1, i2, i3]

    ttt = [tf.gather(controls, ii) for ii in iii]
    
    coeffs = CR_spline_coeffs(*ttt, t = t, dif = dif)
    
    return iii, coeffs
    
    
def get_spline_coeffs( 
        params: List[Tensor], 
        x: Tensor, 
        method = None, 
        derivative:int = None, 
        dtype=None
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
    """
    multidimensional analogue of get_spline_coeffs_1d_simple.
    
    Args:
        params: is a list of len n of 1D tensors. I.e our spline is n-dimensional 
            and params contains for each coordinate the list of corresponding 
            controls of our grid.
        x: is a the point (or batch of points) where we want to evalueate our spline.
            `x.shape = batch_shape + [n]`
        derivative: is a natural number --- the index of the coordinate by which we want to differentiate.
            if it is None, we don't differentiate.
    
    Return: 
        index_list: list containig `4**n` integral tensors of shape `batch_shape`
        multi_index_list: a list of length `4**n` 
            containing integral tensors of shape `batch_shape + [n]`
        coef_list: list containig `4**n` float tensors of shape `batch_shape`
            
    The meaning of result: to calculate value at x, take the grid-point at position 
        multi_index_list[k] with coefficient coef_list[k].
    
    """
    dtype = dtype or K.floatx()
    x = tf.cast(x, dtype)
    
    index_list = [0]
    multi_index_list = [[]]
    coeff_list = [1]
    for k, controls in enumerate(params):
        ind, coef = get_spline_coeffs_1d(
            controls, t = x[..., k], method = method, dif = (derivative == k))
        
        len_controls = tf.cast(tf.shape(controls)[0], tf.int32)
        index_list = [ il * len_controls + i for il in index_list for i in ind]
        multi_index_list = [ ml + [i] for ml in multi_index_list for i in ind]
        coeff_list = [ cl *  c  for cl in coeff_list for c in coef]
        
    multi_index_list = [tf.stack(mi, axis=-1) for mi in multi_index_list]
    return index_list, multi_index_list, coeff_list
    
        
def evaluate_interpolator_one_x(params, values, x, raveled=True, dtype=None):
    """Simplified version of `evaluate_interpolator` for explanatory purposes."""
    assert raveled
    dtype = values.dtype
    x = tf.cast(x, dtype)
    assert len(x.shape) == 1
    
    indices, _, coeffs = get_spline_coeffs(params, x, ravel_multi_index=True)
    
    ret = tf.add_n([ 
        c * tf.gather(values, i) 
        for i, c in zip(indices, coeffs)
    ])   
    
    return ret

    
def evaluate_interpolator(
    params:List[Tensor], 
    values:Tensor, 
    x:Tensor, 
    raveled:bool=True, 
    crop_x:bool=True
)->Tensor:
    """
    Args
        params: is a list of len n of 1D tensors. I.e our spline is n-dimensional 
            and params contains for each coordinate the list of corresponding 
            controls of our grid.
        x: is a the point (or batch of points) where we want to evalueate our spline.
            `x.shape = batch_shape + [n]`    
        values: float tensor. If `raveled==True` its shape is
            `[np.prod([len(c) for c in params])] + codomain_shape`
            otherwise its shape is
            `[len(c) for c in params] + codomain_shape`
        raveled: bool indicating the shape of 'values'
        crop_x: bool if True, we crop `x` to the domain before evaluating
            
    Returns:
        tensor of shape `batch_shape + codomain_shape`
    
    """
    
    if raveled:
        message = "Incompatible shapes, consider changing `raveled` argument."
        tf.assert_equal(values.shape[0], 
                        tf.reduce_prod([len(c) for c in params]), message)
    else:
        message = "Incompatible shapes, consider changing `raveled` argument."
        tf.assert_equal(values.shape[:len(params)], 
                        [len(c) for c in params], message)
        
    floatX = nptf.np_dtype(values)
    x = tf.cast(x, dtype=floatX)
    
    if crop_x:
        min_par = tf.cast(tf.stack([c[0] for c in params]), floatX)
        max_par = tf.cast(tf.stack([c[-1] for c in params]), floatX)
                
        x = tf.maximum(min_par, tf.minimum(x, max_par))
    
    ## see below why we fret so much abou these shapes
    batch_shape = tf.shape(x)[:-1]
    codomain_shape =(tf.shape( values)[1:] if raveled 
                    else tf.shape( values)[len(params):])
    codomain_ndim = tf.size(codomain_shape)
    coeff_new_shape = tf.concat(
        [
            batch_shape, 
            tf.ones([codomain_ndim], dtype = tf.int32)
        ],
        axis = 0
    )
    
    indices, multi_indices, coeffs = get_spline_coeffs(params, x)
    ## now `indices` is an ordinary list of 1D Tensors 
    ## coeffs
    
    if raveled:
        ret = tf.add_n([ 
            tf.reshape(c, coeff_new_shape) * tf.gather(values, i) 
            for i, c in zip(indices, coeffs)
        ])
    else:
        ret = tf.add_n([ 
            tf.reshape(c, coeff_new_shape) * tf.gather_nd(values, mi) 
            for mi, c in zip(multi_indices, coeffs)
        ])
        
    
    return ret        


###
## The rest is untested
#
#
#class InterpolatorEvaluator_tf:
#    """This class does the same job as `evaluate_interpolator`.
#    
#    However when hou have one large batch of `x` and you want to evaluate many
#    interpolators at `x` then it is much more efficient to create one
#    fixed instance of `InterpolatorEvaluator_tf` and then evaluate using that.
#    
#    Attributes:
#        params: list of vectors
#        x: tensor of shape `[n]` or `[batch_len, n]`
#        indices: list of length 4**n of integer tensors of shape `()` or `[batch_len]`
#        coeffs: list of length 4**n of float tensors of shape `()` or `[batch_len]`
#        
#        
#    """
#    def __init__(self, params, x, crop_x = True, name = "interpolator_evaluator"):
#        """
#        Args:
#            params: is a list of len `n` of 1D tensors. Here `n` is the dimension 
#                of the domain of our interpolators.
#            x: a tensor of shape `[batch_len, n]`, or `[n]`. Point(s) where we evaluate.
#            name: string to create tensorflow name_scope
#        """
#        assert len(x.shape) in [1, 2]
#        self.name = name
#        floatX = nptf.np_dtype(x)
#    
#        with tf.name_scope(self.name):
#            if crop_x:
#                min_par = tf.cast(tf.stack([c[0] for c in params]), floatX)
#                max_par = tf.cast(tf.stack([c[-1] for c in params]), floatX)
#                if len(x.shape) == 2:
#                    min_par = min_par[None, :]
#                    max_par = max_par[None, :]
#                x = tf.maximum(min_par, tf.minimum(x, max_par))
#        
#            
#            self.params = params
#            self.x = x
#            self.indices, self.coeffs = get_spline_coeffs(params, x, ravel_multi_index=True)
#        
#    def eval_itp(self, values):
#        """
#        Args:
#            values: tensor containig values of our interpolator at the control points.
#             It has shape `[n_control_pts] + codomain_shape` where 
#             `n_control_pts = np.prod([len(c) for c in params])`
#             
#        Return:
#            a tensor of shape `[batch_len] + codomain_shape` or of `codomain_shape`
#            depending on whether `x` is a batch of points or just one point.
#        """
#        x = self.x
#        
#        codomain_shape = tf.shape( values)[1:]
#        codomain_ndim = tf.size(codomain_shape)
#        coeff_new_shape = tf.concat(
#            [
#                tf.constant([-1], dtype = tf.int32), 
#                tf.ones([codomain_ndim], dtype = tf.int32)
#            ],
#            axis = 0
#        )
#
#        ret = tf.add_n([ 
#            tf.reshape(c, coeff_new_shape) * tf.gather(values, i) 
#            for i, c in zip(self.indices, self.coeffs)
#        ])   
#
#        if len(x.shape) == 1:
#            ret = ret[0]
#            ret.set_shape(values.shape[1:])
#            return ret
#        elif len(x.shape) == 2:
#            ret.set_shape([x.shape[0]] + list(values.shape)[1:] )
#            return ret
#        
#    def __call__(self, values):
#        return self.eval_itp(values)
#    
#    
#class Interpolator_tf:
#    def __init__(self, params, values, raveled = False):
#        self.params = params
#        self.var_ndim = var_ndim = len(params)
#        
#        if raveled:
#            self.raveled_values = values
#        else:
#            self.unraveled_values = values
#            old_shape = tf.shape(values)
#            
#            new_shape = tf.concat(
#                values = [
#                    tf.reduce_prod(old_shape[:var_ndim], keepdims=True),
#                    old_shape[var_ndim:]
#                ], 
#                axis = 0
#            )
#            self.raveled_values = tf.reshape(values, new_shape)
#            
#    def __call__(self, x):
#        return evaluate_interpolator(
#            params = self.params,
#            values = self.raveled_values,
#            x = x,
#            raveled = True
#        )