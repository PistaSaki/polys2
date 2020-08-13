import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import backend as K
from tensorflow.keras.backend import ndim
import itertools as itt
from scipy.special import factorial, binom
from typing import Union, Tuple, List

from polys2.tensor_utils import stack_from_array, apply_tensor_product_of_maps
from .batch_utils import Batched_Object
from .engine import (poly_prod, eval_poly, get_1D_Taylor_matrix, get_1d_Taylor_coef_grid, find_common_dtype)
from .plot_utils import plot_fun


#####################################
## Some other auxilliary funs

def get_bin_indices(bins, tt):
    """ Return bin-indices containing elements of `tt`.
    
    Args:
        bins: float tensor with shape `[noof_bins + 1]`
        tt: float tensor with shape `batch_shape`
        
    Return:
        integer tensor with shape `batch_shape`
    """
    batch_ndim = len(tt.shape)

    tt = tf.cast(tt, bins.dtype)
    tt_reshaped = tt[..., None]
    bins_reshaped = bins[(None,) * batch_ndim + (slice(None),)]
                         
    return tf.reduce_sum(tf.cast(tt_reshaped >= bins_reshaped, tf.int32), axis=-1) - 1
            
####################################
## Classes

class Val_Indexer:
    def __init__(self, obj):
        self.obj = obj
        
    def __getitem__(self, selector):
        return self.obj._val_getitem(selector)
        
        
class Val_Indexed_Object:
    @property
    def val(self):
        return Val_Indexer(obj = self)
    

class Poly(Batched_Object, Val_Indexed_Object):
    def __init__(self, coef, batch_ndim = 0, var_ndim = None, val_ndim = 0):
        self.coef = coef
        self.batch_ndim = batch_ndim
        if var_ndim is None:
            var_ndim = ndim(coef) - batch_ndim - val_ndim
        self.var_ndim = var_ndim
        
        self.val_ndim = val_ndim
        
        try:
            self.degs = tuple(int(d) for d in coef.shape[batch_ndim: batch_ndim + var_ndim])
        except TypeError as err:
            raise Exception(
                err, 
                "Can not infer degrees from coef.shape = {} "
                "batch_ndim = {}, var_ndim = {}, val_ndim = {}".format(
                    coef.shape, self.batch_ndim, self.var_ndim, self.val_ndim
                )
            )

        tf.assert_equal(tf.rank(coef), self.batch_ndim + self.var_ndim + self.val_ndim, message=(
            f"The sum of batch_ndim = {self.batch_ndim}, var_ndim = {self.var_ndim}, val_ndim = {self.val_ndim} "
            f"should be equal to coef_ndim = {tf.rank(coef)}.")
        )
        
    @staticmethod
    def from_tensor(a, batch_ndim=0, var_ndim=None):
        """Generalization of "matrix of quadratic form" -> "polynomial of order 2".
        
        A tensor `a` with `shape = [n]*deg` defines a polynomial of degree `deg` (wrt. each variable) in $R^n$.
        The array of coefficients of the result has shape `[deg]*n`.
        """
        deg = len(a.shape) - batch_ndim
        if deg == 0:
            assert var_ndim is not None, "For a constant polynomial you must specify the number of variables."
            return Poly(coef=a[(...,) + (None,)*var_ndim], batch_ndim=batch_ndim, var_ndim=var_ndim)
        
        n = a.shape[-1]
        if var_ndim is not None: 
            assert n == var_ndim
            
        assert all([d == n for d in a.shape[batch_ndim:]]), f"All the dims should be equal {a.shape[batch_ndim:]}, {a.shape}, {batch_ndim}."

        coef = np.empty(shape=(deg+1, ) * n, dtype=np.object)
        for jj in itt.product(*([range(deg+1)]*n)):
            coef[jj] = tf.zeros(tf.shape(a)[:batch_ndim], a.dtype)
        bs = (slice(None),) * batch_ndim
        for iii in itt.product(*([range(n)]*deg)):
            degs = np.zeros(n, dtype=np.int)
            for i in iii:
                degs[i] += 1

            coef[tuple(degs)] += a[bs + iii]

        coef = stack_from_array(coef, start_index=batch_ndim)
        return Poly(coef, var_ndim=n, batch_ndim=batch_ndim)

    def unit_like(self) -> "Poly":
        """Return a polynomial representing 1 with the same `batch_ndim`, `var_ndim` and `dtype`."""
        return Poly.constant(tf.constant(1, dtype=self.dtype), **self._all_ndims)

    def __repr__(self):
        s = "Poly( " + str(self.coef) 
        if self.batch_ndim > 0:
            s += ", batch_ndim = " + str(self.batch_ndim)
        if self.val_ndim > 0:
            s += ", val_ndim = " + str(self.val_ndim)
        s += " )"
        
        return s
    
    def is_scalar(self):
        return self.val_ndim == 0
    
    def __call__(self, x):
        return eval_poly(self.coef, tf.cast(x, self.dtype), **self._all_ndims)
    
    def cast(self, dtype):
        return Poly(coef=tf.cast(self.coef, dtype), **self._all_ndims)

    @property
    def dtype(self):
        return self.coef.dtype

    def __mul__(self, other, truncation = None):
            
        if isinstance(other, Poly):
            tf.assert_equal(self.batch_ndim, other.batch_ndim)
            tf.assert_equal(self.var_ndim, other.var_ndim)
            # tf.Assert(self.val_ndim == 0 | other.val_ndim == 0 | (self.val_ndim == other.val_ndim),
            #           data=[self.val_ndim, other.val_ndim])
            #           (
            #     "At least one of the two polynomials should have scalar values or both should have the same values. " +
            #     "This can be generalised but is not implemented. "
            # )

            return Poly(
                coef = poly_prod(
                    self.coef, other.coef, 
                    batch_ndim = self.batch_ndim, 
                    var_ndim = self.var_ndim,
                    truncation=truncation
                ),
                batch_ndim = self.batch_ndim,
                var_ndim = self.var_ndim,
                val_ndim = max(self.val_ndim, other.val_ndim)
            )
        else:
            other = tf.convert_to_tensor(other)
            dtype = find_common_dtype([self.dtype], [other.dtype])
            other = tf.cast(other, dtype)
            self = self.cast(dtype)

            other_ndim = ndim(other)
            assert other_ndim <= self.batch_ndim, (
                "You are multiplying a polynomial with batch_dim = " + str(self.batch_ndim) +
                " by a tensor of ndim = " + str(other_ndim) + "."
            )
            other = other[(...,) + (None,) * (self.ndim - other_ndim)]
            
            return Poly(coef=self.coef * other,
                        batch_ndim=self.batch_ndim, var_ndim=self.var_ndim, val_ndim=self.val_ndim)
        
    def __rmul__(self, other):
        return self * other
    
    @property
    def batch_shape(self):
        return tf.shape(self.coef)[:self.batch_ndim]
    
    @property
    def ndim(self):
        return self.batch_ndim + self.var_ndim + self.val_ndim
    
    @property
    def val_shape(self):
        return tf.shape(self.coef)[self.batch_ndim + self.var_ndim:]

    def __add__(self, other):    
        if isinstance(other, Poly):
            tf.assert_equal(self.batch_ndim, other.batch_ndim)
            tf.assert_equal(self.var_ndim, other.var_ndim)
            tf.assert_equal(self.val_ndim, other.val_ndim)

            f, g = self, other
            f = f.pad_degs(g.degs)
            g = g.pad_degs(f.degs)

            return Poly(
                coef=f.coef + g.coef,
                batch_ndim=self.batch_ndim,
                var_ndim=self.var_ndim,
                val_ndim=self.val_ndim
            )
        else:
            other = tf.convert_to_tensor(other)
            dtype = find_common_dtype([self.dtype], [other.dtype])
            other = tf.cast(other, dtype)
            self = self.cast(dtype)
            return self + Poly.constant(other, self.batch_ndim, self.var_ndim, self.val_ndim)
        
    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-1) * other
    
    def __rsub__(self, other):
        return (-1) * self + other
        
    def __truediv__(self, other):
        return 1/other * self

    def val_mul(self, a):
        """
        Multiply values by tensor `a` s.t. `a.ndim == self.val_ndim`.
        """
        assert isinstance(a, (np.ndarray, tf.Tensor, tf.Variable))
            
        f = self
        assert ndim(a) == f.val_ndim
        added_ndim = f.batch_ndim + f.var_ndim
        a = a[(None,) * added_ndim + (Ellipsis, )]
        #print(a.shape, f.coef.shape)
        
        return Poly(
            coef = f.coef * a, 
            batch_ndim = f.batch_ndim,
            var_ndim = f.var_ndim,
            val_ndim = f.val_ndim
        )
        
    def val_sum(self, axis = None):
        """
        For a tensor-valued `Poly` return `Poly` whose values are sums of the values of the original.
        """
        f = self
        if axis is None:
            axis = - np.arange(f.val_ndim) - 1
            axis = tuple(axis)
        else:
            ## make `axis` into a list
            try:
                axis = list(tuple(axis))
            except TypeError:
                axis = (axis, )
                
            ## force the `axis` to be positive
            axis = [i if i >= 0 else i + f.val_ndim for i in axis]
            assert all(0 <= i < f.val_ndim for i in axis )
            
            ##
            axis = np.array(axis) + f.batch_ndim + f.var_ndim
            axis = tuple(axis)
                
        #print("axis =", axis)
        return Poly(
            coef = tf.reduce_sum(
                f.coef, 
                axis = axis
            ),
            batch_ndim = f.batch_ndim,
            var_ndim = f.var_ndim
        )
    
    def truncate_degs(self, limit_degs: Union[int, List[int]]) -> "Poly":
        """Return Poly whose degrees are at most `limit_degs`."""
        degs = tf.minimum(limit_degs, self.degs)
        coef = tf.slice(self.coef, begin=tf.zeros(self.ndim, tf.int32),
                        size=tf.concat([self.batch_shape, degs, self.val_shape], axis=0))
        return Poly(coef=coef, batch_ndim=self.batch_ndim, var_ndim=self.var_ndim, val_ndim=self.val_ndim)

    def pad_degs(self, minimal_degs) -> "Poly":
        """Return Poly whose degrees are at least `minimal_degs`."""
        paddings = tf.concat([
                tf.zeros(self.batch_ndim, dtype=tf.int32),
                tf.maximum(self.degs, minimal_degs) - self.degs,
                tf.zeros(self.val_ndim, dtype=tf.int32),
            ], axis=0)
        
        paddings = tf.stack([tf.zeros_like(paddings), paddings], axis=1)

        return Poly(coef=tf.pad(self.coef, paddings=paddings),
                    batch_ndim=self.batch_ndim, var_ndim=self.var_ndim, val_ndim=self.val_ndim)

    def plot(self, start = None, end = None, **kwargs):
        f = self
        assert f.batch_ndim == 0
        
        start = [0] * f.var_ndim if start is None else start
        end = [1] * f.var_ndim if end is None else end
        plot_fun(f, start, end, **kwargs)
        
    def taylor_at(self, a):
        """Return taylor expansion at `a`."""
        assert a.shape[-1] == self.var_ndim
        if ndim(a) == 1:
            taylor_coef = apply_tensor_product_of_maps(
                matrices=[get_1D_Taylor_matrix(a[..., i], deg=self.degs[i]) for i in range(a.shape[-1])],
                x=self.coef,
                start_index=self.batch_ndim
            )
            return Poly(coef=taylor_coef, **self._all_ndims)
        else:
            raise NotImplementedError()
    
    def shift(self, shift):
        """Return polynomial shifted by `shift`."""
        shift = tf.convert_to_tensor(shift, dtype_hint=K.floatx())
        return self.taylor_at(-shift)
    
    def _get_batch(self, selector):
        coef = self.coef[selector]
        batch_ndim = self.batch_ndim - (len(self.coef.shape) - len(coef.shape))
        assert batch_ndim >= 0
        return Poly(
            coef = coef, 
            batch_ndim = batch_ndim, 
            var_ndim = self.var_ndim, 
            val_ndim = self.val_ndim            
        )

    def _val_getitem(self, selector):
        if not isinstance(selector, tuple):
            selector = (selector,)
            
        coef = self.coef[
            (slice(None), )*(self.batch_ndim + self.var_ndim)
            + selector
        ]
        
        val_ndim = self.val_ndim - (len(self.coef.shape) - len(coef.shape))
        assert val_ndim >= 0, "val_ndim = " + str(val_ndim)
        return Poly(
            coef = coef, 
            batch_ndim = self.batch_ndim, 
            var_ndim = self.var_ndim, 
            val_ndim = val_ndim            
        )

    @property
    def _all_ndims(self):
        return dict(val_ndim=self.val_ndim, var_ndim=self.var_ndim, batch_ndim=self.batch_ndim)
        
    def truncated_exp(self, degs = None):
        """Return exponential in the local ring $R[x]/ x^degs$. 
        
        For example if degs = 4 then `truncated_exp(x) = 1 + x + 1/2 x^2`.
        """
        if degs is None:
            degs = self.degs
        
        g = self
        
        # We divide g into constant `a` and nilpotent part `b` (in the truncated ring)
        a = g.truncate_degs(1)
        b = (g - a).truncate_degs(degs)

        # We compute exp(g) = exp(a + b) = exp(a) * exp(b)
        # `a` is scalar so exp(a) poses no problem.
        exp_a = Poly(
            coef=tf.exp(a.coef),
            batch_ndim=self.batch_ndim,
            var_ndim=self.var_ndim,
            val_ndim=self.val_ndim
        )
        # Since b is nilpotent with highest nonzero power at most n = total degree:
        n = tf.reduce_sum(tf.cast(degs, tf.int32) - 1)
        # we can calculate exp(b) as sum of the first n + 1 terms of the power series.

        exp_b_coef = (1 + b).pad_degs(degs).coef
        b_k_coef = b.pad_degs(degs).coef
        sh = b_k_coef.shape
        factorial_k = 1
        for k in tf.range(2, n+1):
            factorial_k = factorial_k * k
            b_k_coef = poly_prod(b_k_coef, b.coef, truncation=degs,
                                 batch_ndim=self.batch_ndim, var_ndim=self.var_ndim)
            b_k_coef.set_shape(sh)
            exp_b_coef = exp_b_coef + b_k_coef / tf.cast(factorial_k, self.dtype)
        exp_b = Poly(coef=exp_b_coef, **self._all_ndims)
            
        return exp_a * exp_b

        
    def truncated_fun(self, fun_der, degs = None):
        """Return `fun(self)` in the local ring `R[x]/ x^degs` for `fun: R -> R`. 
        
        Generalizes `truncated_exp`.
        
        Args:
            fun_der: callable s.t. `fun_der(k, t)` is a k-th derivative of `fun` at `t \in R`.
                The argument `t` may be a tensor and `fun` should be applied to each coordinate separately.
                The argument `k` is an integer.
            degs: list of ints 
        """
        if degs is None:
            degs = self.degs
        
        g = self
        
        # We divide g into constant `a` and nilpotent part `b` (in the truncated ring)
        a = g.truncate_degs(1)
        b = (g - a).truncate_degs(degs)

        # `b` is nilpotent with highest nonzero power at most n = total degree:
        n = sum([deg - 1 for deg in degs])
        
        # We calculate fun(g) = fun(a + b) as the sum of the first n + 1 terms 
        # of the power series
        # $ \sum_k c_k b^k / k! $
        # where `c_k` is k-th derivative of `fun` at `a`. 

        exp_g = 0
        b_k = 1
        for k in range(n+1):
            c_k = Poly(
                coef = fun_der(k = k, t = a.coef), 
                batch_ndim = self.batch_ndim, 
                var_ndim = self.var_ndim, 
                val_ndim = self.val_ndim 
            )
            
            #print("k = {}; b_k = {}, c_k = {}".format(k, b_k, c_k))
            exp_g = exp_g + c_k * b_k / factorial(k)
            
            b_k = b.__mul__(b_k, truncation = degs)
            
            
        return exp_g
        
    def truncated_inverse(self, degs = None):
        return self.truncated_fun(
            fun_der = lambda k, t: -1 / (-t)**(k+1) * factorial(k),
            degs = degs
        )

    def truncated_power(self, exponent, degs = None):
        """Return `self` raised to the exponent (possibly real) truncated to `degs`. 
        """
        a = exponent
        return self.truncated_fun(
            fun_der = lambda k, t: binom(a, k) * factorial(k) * t**(a-k),
            degs = degs
        )

    def get_taylor_grid(self, params: List[Tensor], truncs: Union[int, Tuple[int]] = None) -> "TaylorGrid":
        from .taylor_grid import TaylorGrid

        assert len(params) == self.var_ndim

        if truncs is None:
            truncs = self.degs
        truncs = truncs * np.ones(len(params), dtype=np.int32)

        taylors = self.coef
        for i, (par, trunc) in enumerate(zip(params, truncs)):
            taylors = get_1d_Taylor_coef_grid(
                coef=taylors,
                poly_index=self.batch_ndim + 2 * i,
                new_index=self.batch_ndim + i,
                control_times=par,
                trunc=trunc
            )

        return TaylorGrid(
            coef=taylors, params = params, 
            batch_ndim=self.batch_ndim, val_ndim=self.val_ndim
        )
    
    def to_PolyMesh(self, params):
        from .poly_mesh import PolyMesh

        assert len(params) <= self.batch_ndim
        return PolyMesh(coef=self.coef, params=params,
                        batch_ndim=self.batch_ndim - len(params), val_ndim=self.val_ndim)

    def to_TaylorGrid(self, params):
        from .taylor_grid import TaylorGrid

        assert len(params) <= self.batch_ndim
        return TaylorGrid(coef = self.coef, params = params,
                          batch_ndim=self.batch_ndim - len(params), val_ndim=self.val_ndim)
        
    def der(self, k = None):
        """Derivative w.r.t. $x_k$."""
        
        if k is None:
            if self.var_ndim == 1:
                k = 0
            else:
                raise Exception("You did not specify by which coordinate you want to differentiate. ")
        
        coef = self.coef
        
        k2 = k + self.batch_ndim 
        deg = self.degs[k]
        ndim = self.ndim
        
        selector = [slice(None)] * ndim
        selector[k2] = slice(1, None)
        selector = tuple(selector)
        
        sh = [1] * ndim
        sh[k2] = deg - 1
        
        coef = coef[selector] * np.arange(1, deg).reshape(sh)
        
        return Poly(
            coef=coef,
            batch_ndim = self.batch_ndim,
            var_ndim= self.var_ndim,
            val_ndim=self.val_ndim
        )

    @staticmethod
    def constant(c, batch_ndim, var_ndim, val_ndim, ):
        tf.assert_equal(ndim(c), 0, "Maybe this method does not do what you want.")
        c = tf.reshape(c, [1] * (batch_ndim + var_ndim + val_ndim))
        return Poly(coef=c, batch_ndim=batch_ndim, var_ndim=var_ndim, val_ndim=val_ndim)
