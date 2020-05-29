""" Autograph problem with overloading operators: `p.__mul__.p` behaves differently from `p * p` in graph mode.

We implement multiplication of polynomials in one variable.
Polynomial $f(x) = a_0 + a_1 x + a_2 x^2 + ... a_n x^n$
is represented by tensor `a = [a_0, a_1,..., a_n]`.
"""
import tensorflow as tf

# The details of the following functions are not important.
def poly_prod(a, b):
    """Return coefficients of product of polynomials with coeffs `a`, `b`."""
    deg_a, deg_b = [tf.shape(x)[0] for x in [a, b]]
    deg_c = deg_a + deg_b - 1

    c_array = tf.TensorArray(dtype=tf.float32, size=deg_c, element_shape=())

    for i in tf.range(deg_a):
        for j in tf.range(deg_b):
            n = i + j
            c_array = c_array.write(n, c_array.read(n) + a[i] * b[j])

    c = c_array.stack()
    return c

# `poly_prod` function works in graph-mode
@tf.function
def square(a):
    return poly_prod(a, a)

print(square(tf.constant([1, 2, 3], dtype=tf.float32)))


# Use `poly_prod` to define multiplication in a class `Poly`
class Poly:
    def __init__(self, coef):
        self.coef = coef

    def __mul__(self, other):
        return Poly(poly_prod(self.coef, other.coef))

# Calculation using `__mul__` works:
@tf.function
def square_2(coef):
    p = Poly(coef)
    return p.__mul__(p).coef

print(square_2(tf.constant([1, 2, 3], dtype=tf.float32)))

# But a code using `*` fails
@tf.function
def square_3(coef):
    p = Poly(coef)
    return (p * p).coef

print(square_3(tf.constant([1, 2, 3], dtype=tf.float32)))

###########
# Workaround: decorating `poly_prod` with `tf.function` removes the problem.