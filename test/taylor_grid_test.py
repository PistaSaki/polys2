import numpy as np
from scipy.stats import norm
import tensorflow as tf
from tensorflow.keras.backend import floatx

from polys2 import Poly
from polys2.taylor_grid import TaylorGrid


def _integrate_gauss(loc, scale, control_points, create_spline: bool):
    params = [control_points]
    gauss_tg = TaylorGrid.from_Gauss_pdf(mu=tf.reshape(loc, [1]), K=tf.reshape(scale ** (-2), [1, 1]),
                                         params=params)
    if create_spline:
        return gauss_tg.get_spline().integrate()
    else:
        return gauss_tg.integrate_spline()


def test_integrating_gauss_distribution():
    control_points = tf.constant([-1.5, 0, 2, 3, 4], floatx())
    loc = tf.constant(2, floatx())
    scale = tf.constant(3, floatx())

    distr = norm(loc, scale)
    exact_integral = distr.cdf(control_points[-1]) - distr.cdf(control_points[0])

    # integrate TaylorGrid
    approx_integral = _integrate_gauss(loc, scale, control_points, create_spline=False)
    print(f"exact_integral = {exact_integral}, approximate_integral = {approx_integral}")
    assert np.allclose(approx_integral, exact_integral, atol=0.001)

    # integrate TaylorGrid in graph-mode
    fun = tf.function(_integrate_gauss)
    approx_integral = fun(loc, scale, control_points, create_spline=False)
    print(f"exact_integral = {exact_integral}, approximate_integral = {approx_integral}")
    assert np.allclose(approx_integral, exact_integral, atol=0.001)

    # integrate Spline
    approx_integral = _integrate_gauss(loc, scale, control_points, create_spline=True)
    print(f"exact_integral = {exact_integral}, approximate_integral = {approx_integral}")
    assert np.allclose(approx_integral, exact_integral, atol=0.001)

    # integrate Spline in graph-mode
    fun = tf.function(_integrate_gauss)
    approx_integral = fun(loc, scale, control_points, create_spline=True)
    print(f"exact_integral = {exact_integral}, approximate_integral = {approx_integral}")
    assert np.allclose(approx_integral, exact_integral, atol=0.001)


def _integrate_poly_wrt_gauss(poly_coef, loc, scale, control_points):
    poly = Poly(poly_coef)
    params = [control_points]
    poly_tg = poly.get_taylor_grid(params)

    gauss_tg = TaylorGrid.from_Gauss_pdf(mu=tf.reshape(loc, [1]), K=tf.reshape(scale**(-2), [1, 1]), params=params)
    product = poly_tg.__mul__(gauss_tg, truncation=2)
    return product.integrate_spline()


def complex_test():
    poly_coef = tf.constant([1, 2, 3], floatx())
    control_points = tf.constant([-1.5, 0, 2, 3, 4], floatx())
    loc = tf.constant(2, floatx())
    scale = tf.constant(2, floatx())

    print(_integrate_poly_wrt_gauss(poly_coef, loc, scale, control_points))
    assert np.allclose(_integrate_poly_wrt_gauss(poly_coef, loc, scale, control_points), 14.233278)

    # in graph-mode
    fun = tf.function(_integrate_poly_wrt_gauss)
    assert np.allclose(fun(poly_coef, loc, scale, control_points), 14.233278)

    fun(poly_coef, loc, scale, control_points)


if __name__ == "__main__":
    test_integrating_gauss_distribution()
    complex_test()
