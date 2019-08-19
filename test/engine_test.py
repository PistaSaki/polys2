import numpy as np
import tensorflow as tf

from polys2.engine import (get_monomials, eval_poly, get_1D_Taylor_matrix,
        get_1d_Taylor_coef_grid, 
        get_1D_Taylors_to_spline_patch_matrix,
        get_1D_integral_of_piecewise_poly
        )
#%%
# test in graph-mode
get_monomials = tf.function(get_monomials) 
eval_poly = tf.function(eval_poly)


#%% 

def test_get_monomials():
    
    mons = get_monomials(
        x = tf.constant([
                [1, 0],
                [0, 2],
                [2, 3]
            ]), 
        degs = [2, 3]
    )
    assert np.allclose(
        mons.numpy(), 
      [[[ 1,  0,  0],
        [ 1,  0,  0]],

       [[ 1,  2,  4],
        [ 0,  0,  0]],

       [[ 1,  3,  9],
        [ 2,  6, 18]]]
    )
 
def test_eval_poly():
    ev = eval_poly(
        coef = tf.constant([2, 0, 1]),
        x = tf.constant([3])
    )
    assert np.allclose(ev, 11)
    
def test_get_1D_Taylor_matrix():
    M = get_1D_Taylor_matrix( a = 2, deg = 4, trunc = 2)
    assert np.allclose( M,
        [[ 1.,  2.,  4.,  8.],
         [ 0.,  1.,  4., 12.]]
    )
    
def test_get_1d_Taylor_coef_grid():
    get_1d_Taylor_coef_grid(
        coef = np.eye(2),
        poly_index = 1,
        new_index = 1, 
        control_times=[0, 1, 2]
    )
    
def test_get_1D_Taylors_to_spline_patch_matrix():
    get_1D_Taylors_to_spline_patch_matrix(0, 1, 2)

def test_get_1D_integral_of_piecewise_poly():
    get_1D_integral_of_piecewise_poly(
        coef = np.array([
            [0, 1, 0],
            [1, 0, 0],
        ]),
        bin_axis = 0,
        polynom_axis = 1,
        control_times = [-1, 0, 1]
    )


    
if __name__ == "__main__":
    test_get_monomials()
    test_eval_poly()
    test_get_1D_Taylor_matrix()

    test_get_1d_Taylor_coef_grid()
    test_get_1D_Taylors_to_spline_patch_matrix()
    test_get_1D_integral_of_piecewise_poly()