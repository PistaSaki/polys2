import numpy as np
import tensorflow as tf
import numbers
import itertools as itt
from scipy.special import factorial, binom

from pitf import ptf, nptf
from pitf.ptf import is_tf_object
import tensor_utils as pit

from batch_utils import Batched_Object
from engine import (array_poly_prod, eval_poly, get_1D_Taylor_matrix,
        get_1d_Taylor_coef_grid)
from plot_utils import plot_fun

#####################################
## Some other auxilliary funs
        
def replace_numbers_in_array(a, zeros):
    for i in range(a.size):
        if isinstance(a.flat[i], numbers.Number):
            a.flat[i] += zeros
            


def get_bin_indices(bins, tt):
    """ Return bin-indices containing elements of `tt`.
    
    Args:
        bins: float tensor with shape `[noof_bins + 1]`
        tt: float tensor with shape `batch_shape`
        
    Return:
        integer tensor with shape `batch_shape`
    """
    batch_ndim = len(tt.shape)

    tt_reshaped = tt[..., None] 
    bins_reshaped = bins[(None,) * batch_ndim + (slice(None),)]
                         
    return tf.reduce_sum(
        tf.cast( tt_reshaped >= bins_reshaped , np.int64),
        axis = -1
    ) - 1
            
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
            var_ndim = coef.ndim - batch_ndim - val_ndim    
        self.var_ndim = var_ndim
        
        self.val_ndim = val_ndim
        
        try:
            self.degs = [int(d) for d in self.coef.shape[batch_ndim: batch_ndim + var_ndim]]
        except TypeError as err:
            raise Exception(
                err, 
                "Can not infer degrees from coef.shape = {} "
                "batch_ndim = {}, var_ndim = {}, val_ndim = {}".format(
                    coef.shape, self.batch_ndim, self.var_ndim, self.val_ndim
                )
            )
            
                     
        assert coef.ndim == self.batch_ndim + self.var_ndim + self.val_ndim,(
            "The sum of batch_ndim = {}, var_ndim = {}, val_ndim = {} "
            "should be equal to coef_ndim = {}.".format(
                self.batch_ndim, self.var_ndim, self.val_ndim, coef.ndim
            )
        )
        
    @staticmethod
    def from_tensor(a, batch_ndim = 0, var_ndim = None):
        """Generalization of "matrix of quadratic form" -> "polynomial of order 2".
        
        A tensor `a` with shape = [n]*deg defines 
        a polynomial of degree `deg` (wrt. each variable) in n-dimensional space.
        The array of coefficients of the result has shape [deg]*n.
        """
        deg = a.ndim - batch_ndim
        if deg == 0:
            assert var_ndim is not None, "For a constant polynomial you must specify the number of variables."
            return Poly(
                coef = a[(...,) + (None,)*var_ndim],
                batch_ndim = batch_ndim,
                var_ndim = var_ndim
            )
        
        a_shape = a.shape
        n = a_shape[-1]
        if var_ndim is not None: 
            assert n == var_ndim
            
        assert all([dim == n for dim in a_shape[batch_ndim:]])
        
        if ptf.is_tf_object(a):
            a_np = ptf.unstack_to_array(a, start_index=batch_ndim)
            f_np = Poly.from_tensor(a_np, batch_ndim=0)
            ## there are some zeros among coeffs of f_np so replace by tf tensors of appropriate shape
            batch_shape = tf.shape(a)[:batch_ndim]
            replace_numbers_in_array(f_np.coef, tf.zeros(batch_shape, dtype = a.dtype))
            
            return f_np._stack_coef_to_tf(batch_ndim = batch_ndim)

        batch_shape = a.shape[:batch_ndim]

        coef = np.zeros(batch_shape + (deg +1,)* n , dtype = a.dtype)
        for iii in itt.product(*([range(n)]*deg)):
            degs = np.zeros(n, dtype = np.int)
            for i in iii:
                degs[i] += 1

            #print(iii, degs, a[iii])    
            bs = (slice(None),) * batch_ndim
            coef[bs + tuple(degs)] += a[bs + iii]

        return Poly(coef, var_ndim = n, batch_ndim=batch_ndim)
    
    def unit_like(p):
        """Return a polynomial representing 1 with the same batch_ndim and degrees as `p`.
        """
        shape = nptf.shape(p.coef)
        batch_ndim = p.batch_ndim
        batch_shape = shape[ : batch_ndim]
    
        dtype = nptf.np_dtype(p.coef)
    
        ## first construct the coefficients for one-element batch
        ## (i.e. batch_ndim = 0)
        one = np.zeros(np.prod(p.degs), dtype)
        one[0] = 1
        one = one.reshape([1]*batch_ndim + p.degs)
    
        ## add the batch-dimensions
        one = nptf.ones(batch_shape, dtype)[(Ellipsis,) + (None,) * p.var_ndim] * one
    
        return Poly(coef = one, batch_ndim= batch_ndim, var_ndim=p.var_ndim)
    
    def _get_data_to_execute(self):
        if ptf.is_tf_object(self.coef):
            return self.coef
        else:
            return []

        
    def _make_copy_from_executed_data(self, data):
        if ptf.is_tf_object(self.coef):
            return Poly(
                coef = data,
                batch_ndim = self.batch_ndim,
                var_ndim = self.var_ndim,
                val_ndim = self.val_ndim,
            )
        else:
            return self
        
        
    def _unstack_coef_to_array(self):
        assert ptf.is_tf_object(self.coef)
        return Poly(
            coef = ptf.unstack_to_array(
                x = self.coef,
                ndim = self.var_ndim,
                start_index = self.batch_ndim
            ),
            batch_ndim = 0,
            var_ndim = self.var_ndim,
            val_ndim = 0
        )
    
    def _stack_coef_to_tf(self, batch_ndim):
        coef = ptf.stack_from_array(
                a = self.coef,
                start_index = batch_ndim            
            )
        
        return Poly(
            coef = coef,
            batch_ndim = batch_ndim,
            var_ndim = self.var_ndim,
            val_ndim = len(coef.shape) - batch_ndim - self.var_ndim
        )
    
    def _put_coefs_to_tf_constant_if_np(self):
        if ptf.is_tf_object(self.coef):
            return self
        else:
            return Poly(
                coef = tf.constant(self.coef),
                batch_ndim = self.batch_ndim,
                var_ndim = self.var_ndim,
                val_ndim = self.val_ndim
            )
        
    
            
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
        if not is_tf_object(x):
            x = np.array(x)
        return eval_poly(self.coef, x, batch_ndim = self.batch_ndim, var_ndim=self.var_ndim, val_ndim=self.val_ndim )
    
    
    
    def __mul__(self, other, truncation = None):
            
        if isinstance(other, Poly):
            assert self.batch_ndim == other.batch_ndim
            assert self.var_ndim == other.var_ndim
            assert self.val_ndim == 0 or other.val_ndim == 0 or (self.val_ndim == other.val_ndim), (
                "At least one of the two polynomials should have scalar values or both should have the same values. " + 
                "This can be generalised but is not implemented. "
            )

            if any([ptf.is_tf_object(f.coef) for f in {self, other}]):
                f, g = [x._put_coefs_to_tf_constant_if_np() for x in [self, other]]
                
                ## It can happen that one of the polynomials 
                ## is not scalar-valued i.e. val_ndim > 0.
                ## In that case, we must add the corresponding number
                ## of dimesions to the values of the other one.
#                print(
#                        "f.val_ndim = " + str(f.val_ndim) + "; " +
#                        "g.val_ndim = " + str(g.val_ndim) + ". " 
#                    )


                if f.val_ndim == 0 or g.val_ndim == 0:
                    if f.val_ndim > g.val_ndim:
                        f, g = g, f
                    f = f.val[(None, ) * (g.val_ndim - f.val_ndim)]
                
                assert f.val_ndim == g.val_ndim, (
                    "f.val_ndim = " + str(f.val_ndim) + "; " +
                    "g.val_ndim = " + str(g.val_ndim) + ". " 
                )
                
                ## make f, g into polys with np.array coef
                f_np, g_np = [x._unstack_coef_to_array() for x in[f, g]]
#                print("shape of elements in coefficient fields of f_np, g_np is " +
#                      str(f_np.coef.flat[0].shape), str(g_np.coef.flat[0].shape)
#                )
                
                prod_np = f_np.__mul__(g_np, truncation = truncation)
                return prod_np._stack_coef_to_tf(self.batch_ndim)
            else:
                return Poly(
                    coef = array_poly_prod(
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
            if isinstance(other, (np.ndarray, tf.Tensor, tf.Variable)):
                other_ndim = nptf.ndim(other)
                assert other_ndim <= self.batch_ndim, (
                    "You are multiplying a polynomial with batch_dim = " + str(self.batch_ndim) +
                    " by a tensor of ndim = " + str(other_ndim) + "."
                )
                other = other[(...,) + (None,) * (self.ndim - other_ndim)]
            
            return Poly(
                coef = self.coef * other,
                batch_ndim = self.batch_ndim,
                var_ndim = self.var_ndim,
                val_ndim = self.val_ndim
            )
        
    def __rmul__(self, other):
        return self * other
    
    @property
    def batch_shape(self):
        return nptf.shape(self.coef)[:self.batch_ndim]
    
    @property
    def ndim(self):
        return self.batch_ndim + self.var_ndim + self.val_ndim
    
    @property
    def val_shape(self):
        return nptf.shape(self.coef)[self.batch_ndim + self.var_ndim:]
    
        
    def __add__(self, other):    
        if isinstance(other, Poly):
            assert self.batch_ndim == other.batch_ndim
            assert self.var_ndim == other.var_ndim
            assert self.val_ndim == other.val_ndim
            
            f, g = self, other
            f = f.pad_degs(g.degs)
            g = g.pad_degs(f.degs)

            return Poly(
                coef = f.coef + g.coef,
                batch_ndim = self.batch_ndim,
                var_ndim = self.var_ndim,
                val_ndim = self.val_ndim
            )
        else:
            return self + Poly.constant(other, self.batch_ndim, self.var_ndim, self.val_ndim)
        
    def __radd__(self, other):
        return self + other
    
        
    def __sub__(self, other):
        return self + (-1) * other
    
    def __rsub__(self, other):
        return (-1) * self + other
        
    def __truediv__(self, other):
        #assert isinstance(other, numbers.Number), "Unsupported type {}.".format(type(other))
        return 1/other * self
        
        
    def val_mul(self, a):
        """
        Multiply values by tensor `a` s.t. `a.ndim == self.val_ndim`.
        """
        assert isinstance(a, (np.ndarray, tf.Tensor, tf.Variable))
            
        f = self
        assert nptf.ndim(a) == f.val_ndim
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
            coef = nptf.reduce_sum(
                f.coef, 
                axis = axis
            ),
            batch_ndim = f.batch_ndim,
            var_ndim = f.var_ndim
        )
    
    def truncate_degs(self, limit_degs):
        degs = np.minimum(limit_degs, self.degs)
        selector = [slice(None)]* self.batch_ndim +  [slice(None, int(deg)) for deg in degs]
        selector = tuple(selector)
        return Poly(
            coef = self.coef[selector], 
            batch_ndim = self.batch_ndim, 
            var_ndim = self.var_ndim, 
            val_ndim = self.val_ndim
        )
    
    
    def pad_degs(self, minimal_degs):
        ## In this first part we can work solely in numpy (i.e. with tangible numbers)
        paddings = np.concatenate([
                np.zeros(self.batch_ndim),
                np.maximum(self.degs, minimal_degs) - self.degs,
                np.zeros(self.val_ndim),
            ]).astype(np.int)
        
        paddings = np.stack([
                nptf.zeros_like(paddings), 
                paddings
        ]).T
        
        #print("paddings =", paddings)
        
        ## Only the padding must be done in appropriate module (np/tf)
        return Poly(
            coef = nptf.pad(self.coef, paddings=paddings), 
            batch_ndim = self.batch_ndim, 
            var_ndim = self.var_ndim, 
            val_ndim = self.val_ndim
        )
        
        
    
    
    
    def plot(self, start = None, end = None, **kwargs):
        f = self
        assert f.batch_ndim == 0
        
        start = [0] * f.var_ndim if start is None else start
        end = [1] * f.var_ndim if end is None else end
        plot_fun(f, start, end, **kwargs)
        
    def Taylor_at(self, a):
        assert a.shape[-1] == self.var_ndim 
        if nptf.ndim(a) == 1:
            taylor_coef = pit.apply_tensor_product_of_maps(
                matrices = [
                    get_1D_Taylor_matrix(ai, deg = self.degs[i])
                    for i, ai in enumerate(a)
                ],
                x = self.coef,
                start_index = self.batch_ndim
            )
            return Poly(
                coef = taylor_coef, 
                batch_ndim = self.batch_ndim, 
                var_ndim = self.var_ndim, 
                val_ndim = self.val_ndim
            )
        else:
            #assert nptf.ndim(a) - 1 = self.batch_ndim
            #for i in range(self.var_ndim):
            #    get_1D_Taylor_matrix(a[..., i], deg = self.degs[i])
            
            raise NotImplementedError()
    
    def shift(self, shift):
        return self.Taylor_at(-np.array(shift))
    
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

        # We conpute exp(g) = exp(a + b) = exp(a) * exp(b)
        # `a` is scalar so exp(a) poses no problem.
        exp_a = Poly(
            coef = nptf.exp(a.coef), 
            batch_ndim = self.batch_ndim, 
            var_ndim = self.var_ndim, 
            val_ndim = self.val_ndim 
        )
        # Since b is nilpotent with highest nonzero power at most n = total degree:
        n = sum([deg - 1 for deg in degs])
        # we can calculate exp(b) as sum of the first n + 1 terms of the power series.

        exp_b = 1 + b
        b_k = b
        for k in range(2, n+1):
            b_k = b_k.__mul__(b, truncation = degs)
            exp_b = exp_b + 1/factorial(k) * b_k
            
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
    
    
      
    def get_Taylor_grid(self, params, truncs = None):
        assert len(params) == self.var_ndim
        assert issubclass( nptf.np_dtype(self.coef).type, np.floating), (
            "Polynomial should have coef of floating dtype. "
            "At the moment dtype = {}.".format(nptf.np_dtype(self.coef))
        )
        
        if truncs is None:
            truncs = self.degs
        if isinstance(truncs, numbers.Number):
            truncs = [truncs]*len(params)
            
            
        taylors = self.coef
        for i, (par, trunc) in enumerate(zip(params, truncs)):
            taylors = get_1d_Taylor_coef_grid(
                coef = taylors, 
                poly_index = self.batch_ndim + 2 * i, 
                new_index = self.batch_ndim + i, 
                control_times = par, 
                trunc = trunc
            )
            
        
        return TaylorGrid(
            coef=taylors, params = params, 
            batch_ndim=self.batch_ndim, val_ndim=self.val_ndim
        )
    
    def to_PolyMesh(self, params):
        assert len(params) <= self.batch_ndim
        return PolyMesh(
            coef = self.coef, 
            params = params, 
            batch_ndim=self.batch_ndim - len(params), 
            val_ndim=self.val_ndim
        )

    def to_TaylorGrid(self, params):
        assert len(params) <= self.batch_ndim
        return TaylorGrid(
            coef = self.coef, 
            params = params, 
            batch_ndim=self.batch_ndim - len(params), 
            val_ndim=self.val_ndim
        )
        
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
        ndim = nptf.ndim(self.coef)
        
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
        if not is_tf_object(c):
            c = np.array(c)
        assert nptf.ndim(c) == 0, "Maybe this method does not do what you want." 
        
        c = nptf.reshape(c, [1]* (batch_ndim + var_ndim + val_ndim))
        return Poly(
            coef = c, 
            batch_ndim = batch_ndim, 
            var_ndim = var_ndim, 
            val_ndim = val_ndim 
        )
        
        
