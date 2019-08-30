from polys2.interpolators import (CR_spline_coeffs, get_spline_coeffs_1d,
    get_spline_coeffs, evaluate_interpolator, Interpolator)
from polys2.tensor_utils import flatten_left

import numpy as np
import tensorflow as tf

#%%
def test_CR_spline_coeffs():
    coeffs = CR_spline_coeffs(t0=0., t1=1., t2=2., t3=3., t = 1.)
    np.allclose(coeffs, [0, 1, 0, 0])
    
    coeffs = CR_spline_coeffs(t0=0., t1=1., t2=2., t3=3., t = [1., 2., 2.])
    assert np.allclose(coeffs, [[0, 0, 0], [1, 0, 0], [0, 1, 1], [0, 0, 0]])

#%%
def test_get_spline_coeffs_1d():
    ## try for one point `t`
    indices, coeffs = get_spline_coeffs_1d(controls=[-1, 0, 1, 2, 3, 4], t=1)
    assert np.allclose(indices, [1, 2, 3, 4])
    assert np.allclose(coeffs, [0, 1, 0, 0])
    
    ## try for batch of points `t`
    indices, coeffs = get_spline_coeffs_1d(controls=[-1, 0, 1, 2, 3, 4], t=[1, 2, 3])
    assert np.array(indices).shape == (4, 3)
    assert np.array(coeffs).shape == (4, 3)
#%%
def test_get_spline_coeffs():
    params = [tf.range(5), tf.range(7)]
    
    ## try with one point `x`
    index_list, multi_index_list, coeff_list = get_spline_coeffs(
            params=params, x = [4.5, 2.5], crop_x=False)
    assert np.array(multi_index_list).shape == (16, 2)
    
    ## try with 1D batch of points
    index_list, multi_index_list, coeff_list = get_spline_coeffs(
            params=params, x=np.ones([10, 2]), crop_x=False)
    assert np.array(multi_index_list).shape == (16, 10, 2)
    
    ## try with 2D batch of points
    index_list, multi_index_list, coeff_list = get_spline_coeffs(
            params=params, x=np.ones([10, 7, 2]), crop_x=False)
    assert np.array(multi_index_list).shape == (16, 10, 7, 2)
         
#%%
def test_evaluate_interpolator():
    ## functions that we interpolate
    def fun(x, y):
        x = tf.convert_to_tensor(x, tf.float32)
        y = tf.convert_to_tensor(y, tf.float32)
        return x**2 - y**2
    
    def fun2(x, y):
        x = tf.convert_to_tensor(x, tf.float32)
        y = tf.convert_to_tensor(y, tf.float32)
        return tf.stack([x**2, y**2], axis=-1)

    ##
    def interpolate_and_evaluate(fun, x, raveled:bool):
        m, n = 5, 7
        params = [tf.range(m), tf.range(n)]
        xx, yy = np.mgrid[:m, :n]
        values = fun(xx, yy)
        if raveled:
            values = flatten_left(values, ndims=2)

        return evaluate_interpolator(params, values, x=x, raveled=raveled) 
    
    
    def check_consistency(fun, x, raveled:bool):
        x = np.array(x)
        interpolated_fun_x = interpolate_and_evaluate(fun, x, raveled)
        fun_x = fun(x[..., 0], x[..., 1])
        assert fun_x.shape == interpolated_fun_x.shape, (
                f"{fun_x.shape} != {interpolated_fun_x.shape}" )
        assert np.allclose(fun_x, interpolated_fun_x)
        
    for raveled in [True, False]:
        ## try with one point `x`
        check_consistency(fun, [1, 2], raveled)
        
        ## try with 1D batch of points 
        check_consistency(fun, [[1, 2], [2, 3], [3, 4]], raveled)
        
        ## try with non-scalar codomain
        check_consistency(fun2, [[1, 2], [2, 3], [3, 4]], raveled)
    
 

#%%
def test_evaluate_interpolator_in_graph_mode():
    def fun(x, y):
        return x**2 - y**2
        
    
    def interpolate_and_evaluate(x):
        m, n = 5, 7
        params = [tf.range(m), tf.range(n)]
        xx, yy = np.mgrid[:m, :n]
        values = fun(xx, yy)
        values = tf.convert_to_tensor(values, tf.float32)
        values = tf.reshape(values, [-1])
        
        return evaluate_interpolator(params, values, x=x)
    
    tf_interpolate_and_evaluate = tf.function(interpolate_and_evaluate)
    
    x = [[1, 2], [2, 3], [3, 4]]
    assert np.allclose(interpolate_and_evaluate(x), 
                       tf_interpolate_and_evaluate(x))


def test_Interpolator():
    def fun(x):
        return tf.stack([tf.sin(x[..., 0]), tf.cos(x[..., 1])], axis=-1)
    
    params = [tf.linspace(0., np.pi, 10), tf.linspace(0., np.pi, 7)]
    
    itp = Interpolator.from_fun(params, fun)
    assert np.allclose(itp([0,0]), [0, 1])
    
def test_Interpolator__dict_values():
    def fun(x):
        return {"sin": tf.sin(x), "cos": tf.cos(x)}
    
    params = [tf.linspace(0., np.pi, 10)]
    
    itp = Interpolator.from_fun(params, fun)
    res = itp([0])
    assert isinstance(res, dict)
    assert np.allclose(res["sin"], 0)
    
    ## try it in graph-mode
    itp = tf.function(itp)
    res = itp([0])
    assert isinstance(res, dict)
    assert np.allclose(res["sin"], 0)


#%%
if __name__ == "__main__":
    test_CR_spline_coeffs()
    test_get_spline_coeffs_1d()
    test_get_spline_coeffs()
    test_evaluate_interpolator()
    test_evaluate_interpolator_in_graph_mode()
    test_Interpolator()
    test_Interpolator__dict_values()
    print("Done.")

#%%
