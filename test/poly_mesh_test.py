import tensorflow as tf
from tensorflow.keras.backend import floatx
import numpy as np

from polys2 import Poly


def test_Spline():
    poly = Poly(tf.constant([2, 3, 4, 5], floatx()))
    taylor_grid = poly.get_taylor_grid(params=[tf.constant([-1, -0.5, 1, 2, 4], floatx())])
    spline = taylor_grid.get_spline()

    x = tf.constant([1.23], floatx())
    assert np.allclose(poly(x), spline(x))
    assert np.allclose(poly.der()(x), spline.der()(x))
    assert np.allclose(spline.integrate(), taylor_grid.integrate_spline())


if __name__ == "__main__":
    test_Spline()
