import tensorflow as tf
from tensorflow.keras.backend import floatx
import numpy as np

from polys2 import Poly


def test_spline():
    poly = Poly(tf.constant([2, 3, 4, 5], floatx()))
    taylor_grid = poly.get_taylor_grid(params=[tf.constant([-1, -0.5, 1, 2, 4], floatx())])
    spline = taylor_grid.get_spline()

    x = tf.constant([1.23], floatx())
    assert np.allclose(poly(x), spline(x))
    assert np.allclose(poly.der()(x), spline.der()(x))
    assert np.allclose(spline.integrate(), taylor_grid.integrate_spline())


def _evaluate_via_spline(poly_coef, control_points, x, der: int = 0):
    poly = Poly(poly_coef)
    taylor_grid = poly.get_taylor_grid(params=[control_points])
    spline = taylor_grid.get_spline()
    for _ in range(der):
        spline = spline.der()
    return spline(x)


def test_spline__in_graph_mode():
    tf_evaluate_via_spline = tf.function(_evaluate_via_spline)
    poly_coef = tf.constant([2, 3, 4, 5], floatx())
    control_points = tf.constant([-1, -0.5, 1, 2, 4], floatx())
    x = tf.constant([1.23], floatx())
    assert np.allclose(_evaluate_via_spline(poly_coef, control_points, x, der=0),
                       tf_evaluate_via_spline(poly_coef, control_points, x, der=0))
    assert np.allclose(_evaluate_via_spline(poly_coef, control_points, x, der=1),
                       tf_evaluate_via_spline(poly_coef, control_points, x, der=1))


if __name__ == "__main__":
    test_spline()
    test_spline__in_graph_mode()

