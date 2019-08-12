import numpy as np
import tensorflow as tf
from poly import Poly

p = Poly( tf.constant([  2.,  30,   0] ))
print(p)
print(p*p)
print(p.truncated_exp())
print(
      p.truncated_fun(lambda k, t: np.exp(t) )
)

print("truncated_inverse(p) = ", p.truncated_inverse())
print("p * truncated_inverse(p) = ", p * p.truncated_inverse())
print("p.der() =", p.der())
