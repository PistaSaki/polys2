import numpy as np
import tensorflow as tf

from polys2.engine import (get_monomials, eval_poly, get_1D_Taylor_matrix, get_1d_Taylor_coef_grid,
                           get_1D_Taylors_to_spline_patch_matrix, get_1d_integral_of_piecewise_poly, poly_prod)


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

    # try in graph mode
    gm = tf.function(get_monomials)
    mons = gm(x=tf.constant([1, 2]), degs=tf.constant([2, 3]))
    assert mons.shape == [2, 3]


def test_eval_poly():
    ev = eval_poly(coef=tf.constant([2, 0, 1]), x=tf.constant([3]))
    assert np.allclose(ev, 11)


def test_get_1D_Taylor_matrix():
    M = get_1D_Taylor_matrix(a=2., deg=4, trunc=2)
    assert np.allclose(M, [[1., 2., 4., 8.],
                           [0., 1., 4., 12.]])


def test_get_1d_Taylor_coef_grid():
    get_1d_Taylor_coef_grid(coef=tf.eye(2), poly_index=1, new_index=1, control_times=tf.constant([0, 1, 2], tf.float32))


def test_get_1d_Taylors_to_spline_patch_matrix():
    get_1D_Taylors_to_spline_patch_matrix(0, 1, 2)


def test_get_1d_integral_of_piecewise_poly():
    get_1d_integral_of_piecewise_poly(
        coef=tf.constant([[0, 1, 0], [1, 0, 0]], tf.float32), bin_axis=0, polynom_axis=1,
        control_times=tf.constant([-1, 0, 1], tf.float32))


def test_poly_prod():
    @tf.function
    def square(a):
        return poly_prod(a, a)

    assert np.allclose(square(tf.constant([1, 2, 3], dtype=tf.float32)), [1, 4, 10, 12, 9])


def test_poly_prod__in_2d():
    a = tf.ones([10, 1, 6])
    b = tf.ones([10, 2, 1])
    c = poly_prod(a, b, batch_ndim=1)
    assert c.shape == [10, 2, 6]

    # nontrivial value shape
    a = tf.ones([10, 1, 6, 3])
    b = tf.ones([10, 2, 1])
    c = poly_prod(a, b, batch_ndim=1, var_ndim=2)
    assert c.shape == [10, 2, 6, 3]


if __name__ == "__main__":
    test_get_monomials()
    test_eval_poly()
    test_get_1D_Taylor_matrix()

    test_get_1d_Taylor_coef_grid()
    test_get_1d_Taylors_to_spline_patch_matrix()
    test_get_1d_integral_of_piecewise_poly()
    test_poly_prod()
    test_poly_prod__in_2d()