import numpy as np
import tensorflow as tf
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
    assert np.allclose(q.unit_like().coef, [1, 0, 0])


def test_poly_multiplication_in_graph_mode():
    @tf.function
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

if __name__ == "__main__":
    test_poly_multiplication_in_graph_mode()
    test_poly_addition_in_graph_mode()
    test_1d_poly()


