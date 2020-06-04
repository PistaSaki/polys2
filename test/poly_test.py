import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from numpy import random as rnd
from polys2.poly import Poly


def test_1d_poly():
    p = Poly(tf.constant([0.,  1,  0]))

    # test __mul__
    assert np.allclose((p*p).coef, [0, 0, 1, 0, 0])

    # test truncated_exp
    exp_p = p.truncated_exp()
    assert np.allclose(exp_p.coef, [1, 1, 1/2])

    # test truncated_fun
    exp_p = p.truncated_fun(lambda k, t: np.exp(t))
    assert np.allclose(exp_p.coef, [1, 1, 1/2])

    # test truncated_inverse
    q = Poly(tf.constant([1.,  1,   0]))
    assert np.allclose(q.truncated_inverse().coef, [1, -1, 1])
    assert np.allclose((q * q.truncated_inverse()).truncate_degs(3).coef, [1, 0, 0])

    # test der
    assert np.allclose(q.der().coef, [1, 0])

    # test unit_like
    assert q.unit_like().coef.shape == [1]
    assert np.allclose(q.unit_like().coef, [1])

def test_2d_poly():
    f = Poly(tf.constant(rnd.randint(-3, 3, size=(10, 1, 6)), K.floatx()), batch_ndim=1)
    g = Poly(tf.constant(rnd.randint(-3, 3, size=(10, 2, 1)), K.floatx()), batch_ndim=1)

    f * g
    assert list((f + g).degs) == [2, 6]
    (f * g).truncate_degs(2)
    f.truncated_exp()

def test_poly_multiplication_in_graph_mode():
    @tf.function#(input_signature=[tf.TensorSpec(None, tf.float32)])
    def poly_square(coef):
        p = Poly(coef)
        return (p * p).coef

    assert np.allclose(poly_square(tf.constant([1, 2, 3], dtype=tf.float32)), [1., 4., 10., 12., 9.])


def test_poly_addition_in_graph_mode():
    @tf.function
    def fun(coef):
        p = Poly(coef)
        return (p + p).coef

    val = fun(tf.constant([1, 2, 3], dtype=tf.float32))
    print(val)
    assert np.allclose(val, [2, 4, 6])


def test_poly_call_in_graph_mode():
    def fun(coef, x):
        p = Poly(coef)
        return p(x)

    fun1 = tf.function(fun)
    val = fun1(tf.constant([1, 0, 1], dtype=tf.float32), tf.constant([2], dtype=tf.float32))
    assert np.allclose(val, 5)

    fun2 = tf.function(fun, input_signature=[tf.TensorSpec([None]), tf.TensorSpec([1])])
    val = fun2(tf.constant([1, 0, 1], dtype=tf.float32), tf.constant([2], dtype=tf.float32))
    assert np.allclose(val, 5)

    # The following does not pass yet:
    # fun3 = tf.function(fun, input_signature=[tf.TensorSpec(None), tf.TensorSpec(None)])
    # val = fun3(tf.constant([1, 0, 1], dtype=tf.float32), tf.constant([2], dtype=tf.float32))
    # assert np.allclose(val, 5)


def test_truncated_exp_in_graph_mode():
    def fun(coef):
        p = Poly(coef)
        return p.truncated_exp().coef

    fun1 = tf.function(fun)
    val = fun1(tf.constant([0., 1, 0]))
    assert np.allclose(val, [1, 1, 1 / 2])

    fun1 = tf.function(fun, input_signature=[tf.TensorSpec([None])])
    val = fun1(tf.constant([0., 1, 0]))
    assert np.allclose(val, [1, 1, 1 / 2])


def test_from_tensor():
    def fun(b, batch_ndim=0):
        p = Poly.from_tensor(b, batch_ndim=batch_ndim)
        return p.coef

    a = tf.constant([[1, 2],
                     [3, 4]], dtype=tf.float32)

    expected_res = tf.constant([[0, 0, 4],
                                [0, 5, 0],
                                [1, 0, 0]])
    assert np.allclose(fun(a), expected_res)

    # try nontrivial batch
    assert np.allclose(fun(a[None, :, :], batch_ndim=1), expected_res[None, :, :])

    # try in graph-mode
    fun1 = tf.function(fun, input_signature=[tf.TensorSpec([2, 2], tf.float32)])
    assert np.allclose(fun1(a), expected_res)

    # try in graph-mode, nontrivial batch
    fun1 = tf.function(fun)
    assert np.allclose(fun1(a[None, :, :], batch_ndim=1), expected_res[None, :, :])


def test_get_taylor_grid():
    def fun(poly_coef):
        p = Poly(poly_coef)
        tg = p.get_taylor_grid(params=[tf.constant([0, 1, 2], dtype=tf.float32)], truncs=2)
        return tg.coef

    poly_coef = tf.constant([0., 1, 2, 3, 4], tf.float32)
    expected_tg_coef = [[0, 1], [10, 30], [98, 173]]

    assert np.allclose(fun(poly_coef), expected_tg_coef)

    # try in graph-mode
    fun1 = tf.function(fun)
    assert np.allclose(fun1(poly_coef), expected_tg_coef)


def test_get_taylor_grid_2d():
    def fun(poly_coef):
        p = Poly(poly_coef)
        tg = p.get_taylor_grid(params=[tf.constant([0, 1, 2], dtype=tf.float32),
                                       tf.constant([0, 1], dtype=tf.float32)],
                               truncs=2)
        return tg.coef

    poly_coef = tf.constant([[0., 1], [2, 3]], tf.float32)

    assert fun(poly_coef).shape == [3, 2, 2, 2]

    # try in graph-mode
    fun1 = tf.function(fun)
    assert fun1(poly_coef).shape == [3, 2, 2, 2]


if __name__ == "__main__":
    test_poly_multiplication_in_graph_mode()
    test_poly_addition_in_graph_mode()
    test_poly_call_in_graph_mode()
    test_truncated_exp_in_graph_mode()
    test_1d_poly()
    test_2d_poly()
    test_from_tensor()
    test_get_taylor_grid()
    test_get_taylor_grid_2d()

    print("Passed!")
